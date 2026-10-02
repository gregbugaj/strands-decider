# Optional multi-image vision

Decider has two checkpoint modes. Existing checkpoints default to `input_mode: text`.
They continue to accept text requests and reject attached images. A newly trained
`multimodal` checkpoint uses the full Qwen3.5 backbone and a pointer head. It accepts
text-only requests, images with optional text, and ordered groups of images. Every
question sees the entire supplied group. A text checkpoint does not gain visual
classification ability just by changing its config.

## Install

From this checkout, with Python 3.10 or newer:

```bash
uv venv
uv pip install -e ".[vision,dev]"
# Add [train] if building the repository's downloaded text corpora.
strands-decider ask --help
strands-decider serve --help
```

Install matching PyTorch and torchvision builds for your CPU/CUDA platform.
See [README-UV.md](../README-UV.md) for the repository's existing installation guidance.
The `vision` extra includes Pillow and torchvision, used by Qwen's processor.
Text inference does not require those optional dependencies.

## Input boundary

Convert TIFFs before calling Decider. Upstream owns TIFF decoding, page extraction,
orientation correction, bit-depth normalization, and page selection. Send each
selected page/crop as a single **8-bit RGB PNG or JPEG**. Decider rejects TIFF bytes,
including TIFF bytes labeled PNG, unsupported formats, grayscale/alpha images,
animated images, invalid base64, and duplicated image IDs. Repeated pixels with
different IDs are allowed and retain their positions.

Images are optional. Omit `images` or pass `[]` for a text-only request. An empty
`state` is valid when images are present. With no images, supply nonempty state text.
The array order is the context order; page numbers do not reorder it. Keep source
IDs and original page numbers when selected pages are nonconsecutive. Per-image
`text` is optional upstream OCR; Decider does not run OCR.

A request to `POST /v1/systemone`:

```json
{
  "state": "Consider both pages together.",
  "images": [
    {
      "id": "page-9",
      "mime_type": "image/png",
      "data_base64": "<base64 of the normalized PNG bytes>",
      "source_id": "document-17",
      "page_number": 9,
      "text": "Optional OCR for page 9"
    },
    {
      "id": "page-1",
      "mime_type": "image/jpeg",
      "data_base64": "<base64 of the normalized JPEG bytes>",
      "source_id": "document-17",
      "page_number": 1
    }
  ],
  "questions": {
    "signed": {"type": "noul", "instructions": "Is there a signature in these pages?"},
    "document_type": {
      "type": "choice",
      "instructions": "Classify the document using both pages.",
      "criteria": {"invoice": "Billing document", "contract": "Agreement"}
    }
  }
}
```

Replace the base64 placeholders with actual bytes. The Python API uses
`engine.ask(state, questions, images=[ImageInput(...)])`; the existing
`engine.ask(state, questions)` call remains supported. Noul responses contain the
yes probability in `noul`, without a separate confidence field.

## CLI and serving

To query images immediately with the cached **pretrained Qwen** and get free-form
text, use the [exact README command](../README.md#query-an-image-now-with-pretrained-qwen).
That example loads Qwen's language head. It does not load or qualify a visual
Decider head; no trained visual Decider checkpoint was produced during implementation.

Use a **trained multimodal checkpoint**:

```bash
strands-decider ask checkpoints/decider-vision-experiment \
  --image pages/page-9.png --image pages/page-1.jpg \
  --noul "Is the document signed?" --json

# The same multimodal checkpoint also accepts requests without images.
strands-decider ask checkpoints/decider-vision-experiment \
  --state "The agreement was signed yesterday." --noul "Was it signed?" --json

strands-decider serve checkpoints/decider-vision-experiment --device cuda
```

CLI images receive ordered IDs `image-1`, `image-2`, etc. Use HTTP/Python to supply
source/page/OCR metadata. For multimodal checkpoints, every question runs a full,
uncached forward. `usage.input_tokens` sums those complete processed sequences;
it includes visual tokens and does not imply shared visual prefill.
Legacy text checkpoints keep the existing prefix cache.

The checkpoint declares image-count, per-image visual-token, aggregate visual-token,
and total sequence budgets. Qwen's saved processor determines resizing/patching
within the declared per-image cap. Actual grid counts and expanded sequence length
are checked; Decider never drops a page or truncates text to fit a visual request.
Over-budget requests fail with actual/allowed counts. Select pages or reduce text
upstream, or train/qualify a larger profile.

`/health` exposes input mode, budgets, transport limits, and whether prefix caching
is effective. Server limits default to 40 MiB full body, 32 MiB base64 image strings,
24 MiB decoded file bytes, 16 million pixels per image, and 32 million pixels total.
Override them with `serve --max-request-bytes`, `--max-image-payload-bytes`,
`--max-decoded-image-bytes`, `--max-pixels-per-image`, and `--max-total-pixels`,
or supply `ImageLimits` to `create_app`. Payload/pixel limits return HTTP 413;
invalid images, incompatible checkpoints, and model token budgets return HTTP 422.

## Dataset and training

Use one question/label per JSONL row. Image paths are relative to that manifest's
directory. Compute SHA-256 over each normalized image file. Example structure:

```json
{
  "kind": "choice",
  "state": "",
  "instructions": "Which document type is shown?",
  "options": [["invoice", "Billing document"], ["contract", "Agreement"]],
  "label": 1,
  "task": "document-type",
  "document_id": "document-17",
  "images": [
    {"id": "page-9", "path": "pages/page-9.png", "sha256": "<64 lowercase hex characters>",
     "source_id": "document-17", "page_number": 9}
  ]
}
```

This is a format illustration, not a labeled training corpus. Supply your real task,
labels, and images. Text-only rows can omit `images`; use `document_id` on related
text rows too. Canonical `label` indexes `options` (noul: false=0, true=1).
Existing option permutation and score targets still apply.

All visual rows require `document_id`. Pages/crops/variants linked by document ID,
source ID, or identical image hash stay in one split. Explicit manifests are checked
for shared keys; filenames cannot infer document families. Group related derivatives
explicitly even if their pixels differ. Assets are hash-checked when loaded.

Copy [configs/vision.yaml](../configs/vision.yaml) into your experiment directory,
set `train_files` and `val_files` to real, document-separated manifests, and export `BASE_REVISION` with the immutable 40-character Hub commit you selected, then run:

```bash
strands-decider train --config configs/my-vision-experiment.yaml \
  --base-revision "$BASE_REVISION"
```

The supplied profile uses at most four images, 1,024 visual tokens per image,
4,096 total visual tokens, and an 8,192-token sequence. These are experimental
configuration values, not measured GPU capacity. It deliberately has no fabricated
base revision or training data and cannot train unchanged.

Training freezes the vision encoder and adapts decoder LoRA plus the fp32 pointer
head. Mixed text/image batches share preprocessing and target construction. Visual
rows receive supervised loss and cannot carry legacy text teacher/replay targets.
If a compatible tied-embedding text reference is enabled, only text rows receive
its KL loss. Precomputed legacy KL, legacy head initialization, and multi-process
multimodal training are currently rejected; this path supports one process/GPU.
The published text recipe is unchanged.

## Qualification and export

Preregister the task, groups/splits, page-selection policy, image limits, hardware,
and success criteria before training. Calibration and evaluation must use documents
unseen in training/validation and separate from each other.

```bash
python evaluation/vision_eval.py checkpoints/decider-vision-experiment \
  --train-manifest data/vision/train.jsonl \
  --validation-manifest data/vision/validation.jsonl \
  --calibration-manifest data/vision/calibration.jsonl \
  --evaluation-manifest data/vision/evaluation.jsonl \
  --fit-calibration --device cuda --out reports/vision/report.json
```

The report includes combined, image-only, and OCR-only conditions, with removed-last-page
and unrelated-last-page substitution controls reported separately. Controls use fixed input order and
document/source/hash-independent donors, never gold answers or model predictions.
A missing donor is recorded as skipped; invalid/over-budget conditions remain failures.
Retained labels may be unanswerable under missing-evidence controls, so their scores
measure sensitivity, not the full-evidence accuracy.

Reports include accuracy, multiclass Brier, top-label ECE, NLL, score MAE in level units,
question-kind/image-count strata, preprocessing/latency/token/memory observations,
checkpoint fingerprint, revision/budgets, and manifest hashes. ECE uses top-label
probability, not API entropy-derived confidence. Calibration can be fitted in memory;
the runner leaves the checkpoint unchanged. To persist calibration, use the existing
`strands-decider calibrate CHECKPOINT --data CALIBRATION_MANIFEST --split all`
on your already independent calibration file, then rerun evaluation. Do not fit or
select pages using evaluation labels. Supplying train/validation manifests checks all
four splits; omitting them leaves training independence unverified in the report.

```bash
python -m strands_decider.hf_export export checkpoints/decider-vision-experiment \
  exports/decider-vision --run-id my-vision-run --reports reports/vision
python -m strands_decider.hf_export verify exports/decider-vision
```

Vision artifacts save the processor, immutable base revision, prompt format, budgets,
adapter, head, and calibration. Export requires processor files and verifies hashes.
The visual model card lists recorded evidence without borrowing text-model results.
Export does not publish to the Hub.

The local tiny-model checks establish processor/backbone integration, gradients,
optional-image handling, and save/reload behavior. They do not establish pretrained
visual accuracy, production memory capacity, or useful confidence thresholds. Qualify
those on your real labeled multi-image documents and target GPU before deployment.

See the [implementation verification record](vision-implementation-status.md) for
executed checks, pretrained runtime smoke evidence, and remaining qualification gates.
