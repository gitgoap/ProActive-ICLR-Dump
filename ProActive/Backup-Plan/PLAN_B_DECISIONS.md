# Plan B frozen decisions

Recorded before selector fitting on 16 September 2026. The owner requested an
independent implementation from the supplied PDF without relying on its missing
ZIP, audit script or packaged cohort CSV. Starting locked test/shift evaluation
still requires the separate `evaluate` command.

The protocol follows the PDF exactly where specified. The PDF's explicitly
unresolved implementation details are frozen as follows:

- Selector score: positive-class `predict_proba`, reproduced from the stored logistic coefficients with the sigmoid function.
- Runtime: the active Python, NumPy and scikit-learn versions are written to `training/environment_preflight.json` before fitting and embedded in the freeze/run manifests.
- Randomness: `numpy.random.default_rng`; image clusters sorted lexicographically before seeded draws.
- Bootstrap percentile interpolation: NumPy `linear` quantiles.
- Floating comparison tolerance: `1e-12`; the required switch margin remains strictly exceeded.
- Invalid bootstrap denominator: redraw; abort after twenty times the requested number of attempts.
- Duplicate-candidate raw provenance: earliest source in `clean, grounding, blur, crop, brightness, noise, blank` order.
- No-candidate training prefix: skip and count. Deployment: abstain and count incorrect.
- Unavailable/non-applicable future probe: no candidate, no replacement and zero logical cost. Attempted invalid/unusable probe: one logical cost, no candidate/vote.

All remaining frozen values—features, logistic settings, weights, acquisition
order, budgets, validation choice, baselines, gates and statistical seeds—are
machine-readable in `config/plan_b_config.json`.
