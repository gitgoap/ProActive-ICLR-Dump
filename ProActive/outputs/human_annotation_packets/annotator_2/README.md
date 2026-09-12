# Short annotator guide

For each row, open the named image and read the question, correct answer, clean
model answer, and probe answers. Fill six cells:

- `clean_correct`: 1 if the clean answer is correct, otherwise 0.
- `visual_fragile`: 1 if visual corruptions make the answer meaningfully less
  stable or correct, otherwise 0.
- `language_persistence`: 1 if the model keeps giving an answer even when the
  image is blank/uninformative, otherwise 0.
- `alignment_instability`: 1 if grounding/relation wording exposes a mismatch
  between the image, question, and answer, otherwise 0.
- `label6`: choose exactly one label below.
- `insufficient_or_contradictory`: 1 when evidence is missing, invalid, or
  genuinely contradictory; otherwise 0.

Allowed `label6` values:

- `no-failure`: no meaningful failure signal.
- `visual`: mainly visual fragility.
- `language-prior`: mainly answer persistence without useful visual evidence.
- `alignment`: mainly image–question–answer mismatch.
- `mixed`: two or more failure signals are present.
- `unclear`: the evidence is too incomplete or contradictory to decide.

Work independently. Do not try to match what you think the automatic system
said. If unsure, use `unclear`; never leave a row blank.
