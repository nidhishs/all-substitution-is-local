# Residual Expertise Is Not Decision Value

When does residual human predictive information actually change the action deployed by an AI system?

We show that residual expertise has decision value only when the human-updated posterior crosses a reward-induced decision boundary.
The paper introduces boundary regret as a deployment estimand for human-AI complementarity and validates it with Lean-formalized finite-action decision theory, synthetic estimator checks, and CheXpert experiments. CIFAR-10H is included as a theory-confirming negative control: on highly accurate classifiers whose softmax probabilities rarely straddle a reward facet, the dissociation signal collapses to near-zero, exactly as the theory predicts.

## Paper

See `paper/` for the full paper source.

## Formal proofs

Lean 4 mechanized proofs for the boundary-regret claims are in `lean/`.

## Code & reproduction

Experiment code and CLI entry points live in `source/`. See `source/data/README.md` for the data pipeline (CheXpert + CIFAR-10H), `source/experiments/README.md` for experiment commands, and `source/experiments/RESULTS.md` for the run-keyed result tables. `source/experiments/run_experiments.sh` runs the full CheXpert pipeline end-to-end.
