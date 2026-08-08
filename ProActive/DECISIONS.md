# Decisions Log

This log records major decisions affecting claims, labels, splits, architecture, cost accounting, calibration, and experimental design.

| Question | Options | Recommended Option | Scientific Effect | Compute Effect | Reversibility | Date | Approved Choice |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Initial architecture and target design | Follow original design vs adopt Super Implementation Plan | Adopt Super Implementation Plan | Focus on Deep Sets and source bits as targets | NA | Yes (re-architecture possible early) | 2026-07-22 | Adopt Super Implementation Plan |
| Week 3 canonical visual severities | Keep preregistered defaults vs select eligible pilot severities | Use pilot-selected non-saturating severities | Fixes the visual intervention strength before full teacher generation | No additional inference cost | Reversible only by invalidating and regenerating downstream teacher caches | 2026-08-09 | Blur `8`; crop `0.65`; brightness `0.15`; noise `25` |
| VizWiz semantic-match threshold | Keep default `0.82` vs calibrate on human-labelled train/validation pairs | Calibrate at target recall `0.90` | Reduces false semantic mismatches for free-form answers while preserving target recall | Negligible CPU cost | Reversible only before full teacher generation | 2026-08-09 | Threshold `0.50` from 50 labels; precision `0.5926`, recall `0.9412`, F1 `0.7273` |
| Week 3 core pilot scope | Wait for InternVL/GQA vs complete the mandatory core with two models/four active datasets | Complete Qwen and Gemma core; retain InternVL/GQA catch-up | Preserves the required two-family validation while avoiding schedule delay | Avoids blocking pilot completion on downloads/data construction | Yes; catch-up artifacts can be added later | 2026-08-09 | Qwen and Gemma over POPE, HallusionBench, VizWiz, and VSR |
