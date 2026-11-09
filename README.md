# Benchmarking-Impossible-Agents

The project tests whether an agent's chain-of-thought adds early warning of verified failures beyond its tool calls and actions.

## Related papers and their metrics

| Paper | Evaluation metrics |
| --- | --- |
| [AgentAbstain: Do LLM Agents Know When Not to Act?](https://arxiv.org/abs/2607.10059) | **Act Accuracy** and **Abstain Accuracy** are the pass rates on their respective variants. **Paired Accuracy** requires both variants to pass. **Conditioned Abstention Rate** measures abstention success among pairs whose act variant passed. Results are macro-averaged across eight categories. |
| [Monitoring Monitorability](https://arxiv.org/abs/2512.18311) | **Monitorability** is the product of sensitivity and specificity (TPR × TNR). The paper compares action-only, chain-of-thought-only, and all-messages monitors. |
| [ImpossibleBench: Measuring LLMs' Propensity of Exploiting Test Cases](https://arxiv.org/abs/2510.20270) | **Cheating rate** is the pass rate on impossible coding variants. The authors also report original-task pass rate and transcript-monitor sensitivity. The flag rate on original tasks is an upper bound on the monitor's false-positive rate. |
| [Agentic Abstention: Do Agents Know When to Stop Instead of Act?](https://arxiv.org/abs/2606.28733) | **AbsRec@K** measures correct abstention by step K. **Timely recall** measures abstention at the earliest warranted step. **AbsRec@10** is overall recall within the interaction budget. **SPL** discounts delayed stops, and **over-abstention rate** tracks stops on solvable tasks. |

## Proposed baseline metrics

The action-only baseline scores intermediate checkpoints using the task, public context, and observed tool calls and actions. A checkpoint is positive when a verified failure occurs within the next five agent decision steps. Checkpoints without a fully observable horizon are handled separately from negatives. Split by task pair so both variants, all runs, and all checkpoints for a pair stay together.

| Metric | What it measures |
| --- | --- |
| **AUROC** | How well scores rank positive checkpoints above negative checkpoints across thresholds. |
| **AUPRC** | Precision–recall performance across thresholds. Report the positive checkpoint fraction as the reference. |
| **Recall at 5% false-positive rate** | Fraction of positive checkpoints flagged at an operating point with 5% false positives among negative checkpoints. |
| **Early-warning lead time** | Decision steps between the first qualifying alert and the verified failure. Report alongside the fraction of failures that received an alert. |
| **False alarms on feasible tasks** | Proposed run-level measure. Fraction of feasible runs with no verified failure that received at least one alert. |

Compare scores before and after evidence of infeasibility appears, break down performance by failure category where counts support it, and estimate uncertainty by bootstrapping task pairs.
