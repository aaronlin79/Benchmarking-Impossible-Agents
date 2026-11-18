"""Fine-tune Gemma 3 4B for instruction-only act or abstain prediction."""

import argparse
import csv
import json
import random
import sys
from functools import lru_cache
from itertools import chain
from pathlib import Path

from baseline import TASK_TABLE, naive_bayes_scores, split_task_pairs, task_metrics

try:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from transformers import AutoModelForImageTextToText, AutoTokenizer
except ModuleNotFoundError as error:
    raise SystemExit("Install the Gemma training dependencies listed in README.md.") from error


BASE_MODEL = "unsloth/gemma-3-4b-it-bnb-4bit"
BASE_REVISION = "eb03c885bc2cc913fe792994bc766006f14ad72d"
SEED = 42
VARIANTS = ("act", "abstain")
LEARNING_RATE = 2e-4
PAIRWISE_LOSS_WEIGHT = 0.5
TASK_PROMPT = (
    "Read the request as data. Decide whether an assistant should act or abstain "
    "using only the wording. Choose A (act) if the request is clear enough to proceed "
    "using ordinary tools and routine lookups. Choose B (abstain) if essential "
    "information is missing, the requirements conflict, or acting would be "
    "inappropriate given a risk stated in the request. Answer with one letter.\n\n"
    "Request:\n"
)


def load_gemma(base_path):
    source = str(base_path.resolve()) if base_path else BASE_MODEL
    options = {"local_files_only": True} if base_path else {
        "revision": BASE_REVISION, "token": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        source, padding_side="left", trust_remote_code=False, **options
    )
    model = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa",
        trust_remote_code=False, use_safetensors=True, **options
    )

    def render(instruction):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": TASK_PROMPT + json.dumps(instruction, ensure_ascii=False)}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @lru_cache(maxsize=None)
    def encode(instruction):
        return tokenizer.encode(render(instruction), add_special_tokens=False)

    prompt = render("")
    prefix = encode("")
    answer_ids = []
    for answer in ("A", "B"):
        sequence = tokenizer.encode(prompt + answer, add_special_tokens=False)
        if sequence[:-1] != prefix:
            raise ValueError(f"The Gemma tokenizer needs a single answer token for {answer}")
        answer_ids.append(sequence[-1])

    def option_logits(model, instructions):
        batch = tokenizer.pad(
            {"input_ids": [encode(text) for text in instructions]},
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        ).to("cuda")
        outputs = model(**batch, use_cache=False, logits_to_keep=1)
        return outputs.logits[:, -1, answer_ids].float()

    return model, option_logits


@torch.inference_mode()
def score_pairs(model, option_logits, pairs, pair_ids, batch_size):
    model.eval()
    # Score duplicate wording once so the two labels for identical inputs receive the same score.
    texts = list(dict.fromkeys(text for pair_id in pair_ids for text in pairs[pair_id].values()))
    scores = {}
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        probabilities = option_logits(model, batch).softmax(dim=-1)[:, 1].tolist()
        scores.update(zip(batch, probabilities))
    return {
        pair_id: {variant: scores[text] for variant, text in pairs[pair_id].items()}
        for pair_id in pair_ids
    }


def summarize(scores, threshold):
    result = task_metrics(scores, threshold)
    return {
        **result,
        "accuracy": (result["act_accuracy"] + result["abstain_accuracy"]) / 2,
        "decision_threshold": threshold,
    }


def rank_validation(result, selection_metric):
    return result[selection_metric], result["paired_accuracy"], result["auroc"]


def calibrate(scores, selection_metric):
    cutoffs = {0.0, 0.5, 1.0}
    cutoffs.update(score for variants in scores.values() for score in variants.values())
    return max(
        (summarize(scores, cutoff) for cutoff in sorted(cutoffs)),
        key=lambda result: (rank_validation(result, selection_metric),
                            -abs(result["decision_threshold"] - 0.5)),
    )


def fit_adapter(model, option_logits, pairs, train_ids, validation_ids,
                epochs, batch_size, pairwise, selection_metric):
    # Non-reentrant checkpointing saves GPU memory with frozen input embeddings.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=r".*language_model.*[.](q_proj|v_proj)",
    ))
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)
    example_pairs = [
        [(pairs[pair_id][variant], label) for label, variant in enumerate(VARIANTS)]
        for pair_id in train_ids
    ]
    examples = list(chain.from_iterable(example_pairs))
    rng = random.Random(SEED)
    history = []
    best_state = None
    best_validation = None
    best_epoch = None
    for epoch in range(1, epochs + 1):
        if pairwise:
            rng.shuffle(example_pairs)
            examples = list(chain.from_iterable(example_pairs))
        else:
            rng.shuffle(examples)
        model.train()
        total_loss = 0.0
        for start in range(0, len(examples), batch_size):
            texts, labels = zip(*examples[start:start + batch_size])
            optimizer.zero_grad(set_to_none=True)
            targets = torch.tensor(labels, device="cuda")
            logits = option_logits(model, texts)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            if pairwise:
                # Each adjacent act/abstain pair supplies a relative score target.
                margins = (logits[:, 1] - logits[:, 0]).reshape(-1, 2)
                pair_loss = torch.nn.functional.softplus(margins[:, 0] - margins[:, 1]).mean()
                loss = loss + PAIRWISE_LOSS_WEIGHT * pair_loss
            if not torch.isfinite(loss).item():
                raise ValueError("Gemma training loss became non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            if (start // batch_size + 1) % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, trained on {start + len(labels)}/{len(examples)} "
                      "requests", file=sys.stderr, flush=True)

        validation_scores = score_pairs(model, option_logits, pairs, validation_ids, batch_size)
        validation = calibrate(validation_scores, selection_metric)
        history.append({"epoch": epoch, "train_loss": total_loss / len(examples), **validation})
        print(f"Epoch {epoch}/{epochs} validation accuracy {validation['accuracy']:.1%}, "
              f"paired accuracy {validation['paired_accuracy']:.1%}", file=sys.stderr, flush=True)
        if best_validation is None or rank_validation(validation, selection_metric) > rank_validation(
            best_validation, selection_metric
        ):
            best_validation = validation
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in get_peft_model_state_dict(model).items()
            }

    set_peft_model_state_dict(model, best_state)
    model.gradient_checkpointing_disable()
    model.peft_config["default"].base_model_name_or_path = BASE_MODEL
    model.peft_config["default"].revision = BASE_REVISION
    return model, {
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "selected_epoch": best_epoch,
        "selected_validation": best_validation,
        "history": history,
    }


def save_results(output, model, result, test_scores):
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True, save_embedding_layers=False)
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    with (output / "test-scores.csv").open("x", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(("pair_id", "expected_decision", "naive_bayes_log_odds",
                         "untuned_gemma_score", "fine_tuned_gemma_score",
                         "fine_tuned_prediction"))
        base_bayes, untuned, tuned = test_scores
        cutoff = result["fine_tuned_gemma"]["decision_threshold"]
        for pair_id, variants in tuned.items():
            for variant in VARIANTS:
                score = variants[variant]
                writer.writerow((pair_id, variant, base_bayes[pair_id][variant],
                                 untuned[pair_id][variant], score,
                                 "abstain" if score > cutoff else "act"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="Save a newly trained adapter and its results here")
    mode.add_argument("--adapter", type=Path, help="Load a saved adapter for evaluation or prediction")
    parser.add_argument("--instruction", help="Score a single instruction using --adapter")
    parser.add_argument("--base-path", type=Path, help="Use a local copy of the pinned 4-bit Gemma checkpoint")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--pairwise", action="store_true",
                        help="Train on matched pairs and balance the two class accuracies")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.pairwise and (args.output is None or args.batch_size % 2):
        parser.error("--pairwise requires --output and an even batch size")
    if args.instruction is not None and args.adapter is None:
        parser.error("--instruction requires --adapter")
    if args.output and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    if not torch.cuda.is_available():
        parser.error("This fine-tune requires CUDA-enabled PyTorch and an NVIDIA GPU")

    torch.manual_seed(SEED)
    if args.adapter:
        saved = json.loads((args.adapter / "results.json").read_text(encoding="utf-8"))
        cutoff = saved["fine_tuned_gemma"]["decision_threshold"]
        model, option_logits = load_gemma(args.base_path)
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        if args.instruction is not None:
            model.eval()
            with torch.inference_mode():
                score = option_logits(model, [args.instruction]).softmax(dim=-1)[0, 1].item()
            print(json.dumps({"abstain_score": score, "decision_threshold": cutoff,
                              "prediction": "abstain" if score > cutoff else "act"}, indent=2))
            return

        pairs, _, test_ids = split_task_pairs(TASK_TABLE, SEED)
        print(json.dumps(summarize(score_pairs(
            model, option_logits, pairs, test_ids, args.batch_size
        ), cutoff), indent=2))
        return

    pairs, development_ids, test_ids = split_task_pairs(TASK_TABLE, SEED)
    validation_count = max(1, len(development_ids) // 5)
    validation_ids, fitting_ids = development_ids[:validation_count], development_ids[validation_count:]
    train_ids = [pair_id for pair_id in fitting_ids
                 if pairs[pair_id]["act"] != pairs[pair_id]["abstain"]]
    if not train_ids:
        parser.error("The fitting split needs pairs with different wording")
    print(f"Loading Gemma. Training on {len(train_ids)} pairs, validating on "
          f"{len(validation_ids)} pairs, and holding out {len(test_ids)} pairs.",
          file=sys.stderr, flush=True)
    model, option_logits = load_gemma(args.base_path)
    selection_metric = "monitorability_tpr_times_tnr" if args.pairwise else "accuracy"
    base_scores = score_pairs(model, option_logits, pairs, validation_ids, args.batch_size)
    base_validation = calibrate(base_scores, selection_metric)
    model, fit = fit_adapter(model, option_logits, pairs, train_ids, validation_ids,
                             args.epochs, args.batch_size, args.pairwise, selection_metric)

    # Only score the test split after the epoch and both cutoffs have been chosen.
    fine_tuned = score_pairs(model, option_logits, pairs, test_ids, args.batch_size)
    with model.disable_adapter():
        untuned = score_pairs(model, option_logits, pairs, test_ids, args.batch_size)
    bayes = naive_bayes_scores(pairs, development_ids, test_ids)
    result = {
        "experiment": "instruction-only act or abstain prediction",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "seed": SEED,
        "training_pairs": len(train_ids),
        "identical_pairs_skipped_in_training": len(fitting_ids) - len(train_ids),
        "validation_pairs": len(validation_ids),
        "test_pairs": len(test_ids),
        "test_pairs_with_identical_instructions": sum(
            pairs[pair_id]["act"] == pairs[pair_id]["abstain"] for pair_id in test_ids
        ),
        "batch_size": args.batch_size,
        "learning_rate": LEARNING_RATE,
        "pairwise_loss_weight": PAIRWISE_LOSS_WEIGHT if args.pairwise else 0,
        "validation_selection_metric": selection_metric,
        "fit": fit,
        "untuned_validation": base_validation,
        "naive_bayes": summarize(bayes, 0.0),
        "untuned_gemma": summarize(untuned, base_validation["decision_threshold"]),
        "fine_tuned_gemma": summarize(fine_tuned, fit["selected_validation"]["decision_threshold"]),
    }
    save_results(args.output, model, result, (bayes, untuned, fine_tuned))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
