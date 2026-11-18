# Gemma findings

Gemma performed best after training on matched pairs. At a limit of 5% false alarms, it caught 12 requests that called for pausing, compared with six for naive Bayes.

## Same test set

Every run used the request text from the same 52 held-out AgentAbstain pairs (104 requests, split seed 42). Each pair has a should-act and a should-pause version. [Task data](data/agentabstain-tasks.jsonl)

| Run | Accuracy | AUROC | Avg. precision | Both versions right | Pause recall at up to 5% false alarms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Naive Bayes | 55.8% | 0.582 | 0.576 | 6/52 | 11.5% |
| Gemma, original Ollama prompt | 50.0% | 0.460 | 0.481 | 0/52 | 0.0% |
| Gemma, two-choice scoring before training | 53.8% | 0.545 | 0.563 | 4/52 | 7.7% |
| Gemma with LoRA | 56.7% | 0.616 | 0.634 | 7/52 | 17.3% |
| **Gemma with matched-pair LoRA** | **58.7%** | **0.664** | **0.669** | **9/52** | **23.1%** |

## What helped

- **Score two fixed answers.** The original prompt predicted pause for all 104 requests. Scoring the two choices directly and choosing a cutoff on validation gave Gemma a better starting point.
- **Train a small adapter.** LoRA improved accuracy and ranking over the same 4-bit model before training. Both fine-tunes used 85 pairs with different wording and skipped 84 fitting pairs with identical text and opposite labels.
- **Learn from the difference within each pair.** The best run penalized Gemma when it ranked the should-act version above its should-pause twin. It used 42 validation pairs to choose the epoch and cutoff, rewarding both act and pause accuracy. Each test request received its own score. [Run instructions](README.md#fine-tune-gemma-on-the-task-table)

## Limits

At its selected cutoff, matched-pair Gemma got 44 of 52 act requests and 17 of 52 pause requests right. Naive Bayes got 32 and 26, respectively. The overall gain was three requests (95% paired bootstrap interval of -1.0 to +6.7 percentage points). In 27 test pairs, the wording is identical despite opposite labels. Early-warning performance requires labeled agent runs.
