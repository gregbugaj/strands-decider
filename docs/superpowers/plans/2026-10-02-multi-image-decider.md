# Multi-image Decider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task inline. Steps use checkbox (`- [ ]`) syntax for tracking. The proposed execution method is subject to the user's plan review.

**Goal:** Implement training and inference for ordered groups of normalized page images using Qwen's multimodal backbone and Decider's pointer readout.

**Architecture:** Keep the legacy text route and add a multimodal checkpoint mode. A shared preparation module validates normalized images, invokes the Qwen processor, and maps option positions into the expanded token sequence. Vision requests use independent, uncached question forwards initially.

**Tech Stack:** Python >=3.10, PyTorch >=2.7, Transformers >=5.15,<6, PEFT >=0.21, Pydantic/FastAPI/Typer, Pillow, pytest/Ruff/mypy, setuptools/setuptools-scm.

**Spec:** [Approved multi-image design](../specs/2026-10-02-multi-image-decider-design.md).

## Global Constraints

- TIFF decoding, page extraction, orientation, bit-depth normalization, and page selection happen upstream. Decider accepts normalized single-image PNG/JPEG only.
- Image groups are ordered joint context; never silently discard images or substitute independent page predictions.
- Preserve legacy checkpoints, text request behavior, option permutation, yes/no probability output, score semantics, and the published v19 recipe.
- `input_mode` is `text` or `multimodal`; missing mode on a legacy checkpoint means `text`.
- Multimodal checkpoints declare positive `max_images`, `max_visual_tokens_per_image`, `max_total_visual_tokens`, and `max_length` values. Transport limits are explicit serving configuration.
- The first multimodal inference implementation bypasses prefix caching. Vision-cache optimization is a later deliverable, not part of this plan.
- Freeze vision weights; train decoder LoRA and the fp32 pointer head. Text-only teacher/reference objectives do not apply to visual rows.
- Pin the base model to an immutable revision for multimodal artifacts. Save the processor, prompt format, budgets, adapter, head, and calibration.
- Keep Python 3.10 compatibility; do not adopt Python-3.12-only syntax or change package compatibility to match the local interpreter.
- Preserve existing README changes, README-UV.md, start-codex.sh, and unrelated files. An overall uv migration is outside this feature.
- Cloud hosts, paid labeling, and publication require their own authorization under training/AGENTS.md. Local code checks do not qualify production model quality.

## Review Focus

- A PNG/JPEG MIME label with TIFF bytes must be rejected by decoded format, not accepted by extension or header alone (Task 1).
- Duplicate image content with different identifiers is legitimate and must retain both positions; duplicated identifiers fail (Tasks 1 and 2).
- Variable image counts and resolutions in one training batch must preserve sample boundaries and option indices (Task 5).
- An image request sent to a legacy text checkpoint must fail clearly instead of falling back to text (Tasks 3 and 4).
- A processor or base revision missing from a visual artifact must fail before reconstruction can silently use changing Hub metadata (Tasks 3 and 6).

## File Responsibilities

Create `src/strands_decider/vision.py` for transport decoding, normalized image records, budget checks, and multimodal preparation. Keep neural-network loading/forwarding in `modeling.py`; keep supervised target construction in `data/collate.py`. Create `evaluation/vision_eval.py` for visual ablations, calibration breakdowns, and resource reports. Add focused vision tests beside existing legacy tests, a separate visual training configuration, and `docs/vision.md` for the input contract and qualification commands.

## Task 1: Normalized image contract and input validation

**Files:** Modify `src/strands_decider/schema.py`, `pyproject.toml`; create `src/strands_decider/vision.py`, `tests/test_vision_inputs.py`; extend `tests/test_core.py` where legacy validation is covered.

**Interfaces:**
- `ImageInput(BaseModel)`: `id: str`, `mime_type: Literal['image/png', 'image/jpeg']`, `data_base64: str`, `source_id: str | None`, `page_number: int | None`, `text: str | None`.
- `SystemOneRequest.images: list[ImageInput]` defaults to an empty list.
- `ImageLimits`: positive `max_payload_bytes`, `max_decoded_bytes`, `max_pixels_per_image`, `max_total_pixels`; explicit values supplied by the serving configuration.
- `DecodedImage`: validated image pixels and the original `ImageInput` metadata.
- `decode_images(images: Sequence[ImageInput], limits: ImageLimits) -> list[DecodedImage]` preserves order.
- `ImagePayloadTooLarge(ValueError)` distinguishes HTTP 413 from ordinary HTTP 422 errors.

- [ ] Write tests: `test_image_only_request_accepts_empty_state` accepts `state=''` with a valid image; `test_empty_request_still_fails` rejects empty state with no images; `test_duplicate_ids_fail` rejects repeated identifiers; `test_duplicate_pixels_preserve_order` retains differently named copies.
- [ ] Write decoding tests using generated RGB PNG/JPEG fixtures: reject malformed base64, TIFF bytes labeled PNG, non-RGB/animated images, empty IDs, nonpositive page numbers, MIME/decoded-format mismatch, and exceeded byte/pixel limits. Treat identifiers containing special-token text as ordinary escaped metadata, not prompt structure.
- [ ] Run `pytest -q tests/test_vision_inputs.py`; confirm failures are missing behavior, then implement validation/decoding and add a `vision` extra with Pillow plus its typing support in the relevant development/lint setup.
- [ ] Rerun the focused tests and legacy schema tests. Record results in this plan before moving on. A commit, if requested, contains only this task's files and runs enabled hooks.

## Task 2: Shared multimodal preparation and pointer indexing

**Files:** Extend `vision.py`; modify `prompting.py`; create `tests/test_vision_preparation.py`.

**Interfaces:**
- `VisionBudget`: required positive `max_images`, `max_visual_tokens_per_image`, `max_total_visual_tokens`, `max_length`.
- `PreparedVisionBatch`: processor tensors, padded `opt_idx`, answer positions, per-example visual/sequence token counts, and ordered source/image metadata.
- `prepare_vision_batch(processor: Any, states: Sequence[Content], images: Sequence[Sequence[DecodedImage]], questions: Sequence[RenderedQuestion], budget: VisionBudget) -> PreparedVisionBatch`.
- Preparation uses the checkpoint's `prompt_format='vision-v1'`; global state comes first, then each image's escaped identifiers/page metadata, image segment, and OCR, followed by the existing question/options/answer suffix.

- [ ] Write tests with the real Qwen processor/tokenizer when locally available and a deterministic processor fixture for ordinary offline CI. For image groups of lengths one and three, assert source order, exact option-token text, surviving answer marker, and correctly padded positions. The fixture is an orchestration aid; real tiny-model tests in Task 3 remain required.
- [ ] Write tests for nonconsecutive page numbers, repeated option wording, option permutation, differing resolutions, and text-only rows within a multimodal batch. Assert exact-boundary budgets pass and over-limit image, per-image visual, aggregate visual, and complete sequence counts raise errors containing actual/allowed counts.
- [ ] Run `pytest -q tests/test_vision_preparation.py` and observe missing behavior. Implement shared preparation using official processor expansion and metadata; never construct pixels/patch embeddings manually.
- [ ] Map original tokenizer-offset positions into processor-expanded IDs by validating non-image token alignment and accounting for each image-placeholder expansion using actual grids and processor merge size. Reject mismatches instead of guessing. Verify option and answer positions against the complete final sequence. Do not concatenate independently tokenized segments without proving alignment.
- [ ] Right-pad heterogeneous examples without mixing flattened image tensors across samples; keep integer metadata integer when moving inputs to a device. Count actual tokens and reject overlength requests without truncation. Rerun preparation tests.

## Task 3: Full Qwen multimodal model and reproducible reconstruction

**Files:** Modify `modeling.py`; create `tests/test_vision_model.py`; extend `tests/test_checkpoint_load.py` and `tests/test_hybrid.py`.

**Interfaces:**
- Extend `StrandsDeciderConfig` with input mode, immutable `base_revision`, prompt format, and optional visual-budget fields required only for multimodal mode.
- Extend `encode(...)` and `forward(...)` with optional `pixel_values`, `image_grid_thw`, `mm_token_type_ids`, `position_ids`, and explicit answer positions. Legacy callers retain their existing arguments.
- Store `model.processor` for multimodal checkpoints; `model.tokenizer` remains available as the processor's tokenizer. Legacy models need no processor.

- [ ] Write tests that legacy JSON selects text mode and multimodal JSON without budgets/revision fails early. Write a tiny, randomly initialized full Qwen3.5 model test with actual processor-shaped image tensors: assert `[batch, option_count]` logits, finite loss, and gradients through decoder adapters/head with frozen vision parameters.
- [ ] Observe failures using `pytest -q tests/test_vision_model.py tests/test_checkpoint_load.py tests/test_hybrid.py`.
- [ ] Load `Qwen3_5Model` with the full config and pinned revision in multimodal mode. Retain the old text loader. Read full-config hidden size through `text_config`; target only exact decoder module paths for LoRA, including recurrent projections; verify no vision parameter becomes trainable.
- [ ] Forward multimodal tensors through the backbone, gather explicitly verified answer/option positions, and keep the head fp32. Verify device/dtype movement on CPU and GPU-compatible paths. Reject vision with a text checkpoint.
- [ ] Save/load processor files with vision checkpoints, propagate the revision consistently to config/model/processor loads, and reject absent required artifacts. Validate save/reload prediction equality within declared tolerances and legacy fixture loading.

## Task 4: Uncached joint-image inference, CLI, and HTTP

**Files:** Modify `infer.py`, `cli.py`, `server.py`; create `tests/test_vision_infer.py`; extend `tests/test_server.py` and add `tests/test_vision_cli.py`.

**Interfaces:**
- `SystemOneEngine.evaluate(request: SystemOneRequest) -> SystemOneResponse` dispatches on checkpoint input mode/request images. `ask(state: Content, questions: dict[str, Question], *, images: list[ImageInput] | None = None)` retains positional compatibility.
- `ask` CLI gains repeatable `--image`; absent state becomes an empty string only when image input exists. Generated IDs follow argument order (`image-1`, `image-2`, ...).
- Serving config exposes image transport limits; health exposes input mode and model budgets. Vision requests use one full multimodal forward per question and bypass shared-prefix caching.

- [ ] Write tests for image-only CLI calls, three repeated image arguments, HTTP source/page/OCR metadata, joint-context propagation into every question, 422 for TIFF or unsupported checkpoints, and 413 for oversized payloads. Preserve legacy CLI required-state errors when no images are supplied.
- [ ] Test multi-question requests with cache enabled globally: assert the vision branch does not enter the text cache, and text requests on legacy checkpoints still do. Verify yes/no responses have `noul` and no invented confidence field.
- [ ] Run focused tests, observe failures, then implement dispatch using Task 2 preparation and Task 3 forwarding. Keep one shared answer-readback implementation for temperatures, scores, and dynamic label names.
- [ ] Count complete processed input sequences across uncached question forwards in `usage.input_tokens`; avoid claiming one-time prefill reuse. Run legacy server/CLI/cache tests and the new vision tests.

## Task 5: Training records, batching, objectives, and grouped splits

**Files:** Modify `data/format.py`, `data/collate.py`, `data/sampling.py`, `train.py`, `evaluate.py`; create `tests/test_vision_training.py`.

**Interfaces:**
- `ImageAsset`: `id`, normalized PNG/JPEG `path`, `sha256`, optional source/page/OCR metadata. Example records gain `images` (empty by default) and optional `document_id` for split grouping.
- `read_jsonl(path)` resolves asset paths relative to the manifest directory without changing serialized relative paths; decoder verifies asset hashes.
- A multimodal collator consumes the model processor/budget and existing target/permutation machinery, returning model tensors, `visual_rows` eligibility flags, and supervised labels/weights. Dispatch through `SystemOneCollator` or a sibling `VisionCollator` without copying target logic.
- `TrainConfig` gains multimodal config fields; model-forward argument construction is shared by training and validation. Visual rows are excluded from legacy reference-KL and replay/teacher terms, with eligible counts logged.
- Document-grouped splitting is deterministic by seed. Explicit train/validation/calibration/evaluation manifests reject shared document/source group keys. Dataset manifests explicitly group linked crops/variants; filenames alone cannot infer this.

- [ ] Write tests for legacy JSONL readback, manifest-relative image paths, wrong hashes, a mixed batch with zero/one/three images, label remapping, finite supervised gradients, and frozen vision weights. Assert processor flattening preserves image/sample associations.
- [ ] Write tests that document pages and related variants share one split; held-out manifests with overlapping document groups fail. Test text KL stays enabled while visual rows have no legacy KL/teacher loss; reject accidental visual replay attachment.
- [ ] Run failing tests, then implement asset resolution, shared preparation, multimodal forwarding in both training and validation, group splitting, and objective masks. Length grouping must include actual or validated estimated visual-token cost rather than text character length alone.
- [ ] Implement image-aware evaluation/calibration example conversion so image fields reach `engine.evaluate`; do not lose them via the current text-only `ask` call. Run focused training and existing corpus identity/sampling/score tests.

## Task 6: Honest multimodal export and artifact verification

**Files:** Modify `hf_export.py`; extend `tests/test_hf_export.py` with multimodal fixtures.

**Interfaces:**
- Export dispatches on `input_mode`; legacy v19 exports retain their current format and claims.
- Vision exports copy the complete saved processor assets, adapter, head, config, and qualification records. Their card/provenance use the configured base revision, budgets, and measured visual results rather than hardcoded v19 text benchmarks.
- Verification requires all mode-specific artifacts and checks their hashes; missing processor/revision cannot be filled from a mutable Hub default.

- [ ] Write tests that a visual export includes processor assets, immutable revision, budgets, and actual evaluation metadata. Assert no v19 accuracy/latency or visual confidence claims appear without supporting records. Missing required processor files fail even when the general manifest is valid.
- [ ] Run failing export tests; implement mode-aware copy/verification/card/provenance while preserving legacy behavior and safe export overlap checks.
- [ ] Verify deterministic repeated export, manifest readback, save/reload prediction agreement, and all existing export tests.

## Task 7: Visual experiment, evaluation, and user documentation

**Files:** Create `configs/vision.yaml`, `evaluation/vision_eval.py`, `docs/vision.md`, `tests/test_vision_eval.py`; update `training/README.md`, `evaluation/README.md`, and add a narrow link in `README.md`.

**Interfaces:**
- `vision_eval.py` accepts checkpoint and independent evaluation/calibration manifest paths explicitly. Report image-only, OCR-only, combined, removed-page, and unrelated-page conditions; stratify by question kind and image count, recording failures as well as successes.
- Reports record accuracy, Brier/ECE where applicable, score errors, preprocessing time, total latency, token counts, peak GPU memory, revision, budgets, and source manifest hashes.
- `configs/vision.yaml` is a new experiment configuration with explicit budgets, document-group metadata requirements, frozen vision encoder, decoder LoRA, supervised visual objectives, and a separate output directory. Its limits are labeled an experimental profile until hardware qualification.

- [ ] Write report tests against known labels/probabilities: accuracy 1.0 for correct choices, missing-evidence controls accounted separately, failure counts retained, calibration split overlap rejected, and no outcome-dependent page selection.
- [ ] Run failing report tests; implement evaluation/ablation and calibration reporting, retaining existing per-question temperature semantics. Write instructions for creating grouped manifests and upstream RGB PNG/JPEG conversion, repeatable CLI images, HTTP bytes, and explicit budget errors.
- [ ] Keep the published v19 configuration unchanged. Require actual task labels, independent splits, a pinned revision, and a preregistered experiment before training; do not create fabricated training rows, taxonomy labels, or empirical success claims to populate the config.
- [ ] Verify examples' shell/JSON syntax, documentation links, CLI help, report tests, and the new configuration's parsing/validation behavior.

## Task 8: Whole-change verification and model qualification handoff

**Files:** Update this plan's execution notes; add qualification records only for checks actually executed.

- [ ] At execution start, inspect workspace isolation and existing environments. Arrange an isolated workspace only with the applicable worktree workflow, preserve local changes, install the relevant extras into a dedicated environment, and run baseline `HF_HUB_OFFLINE=1 pytest -q`.
- [ ] After Tasks 1-7 pass individually, run `HF_HUB_OFFLINE=1 pytest -q`, `ruff check .`, `ruff format --check` on changed Python files, `mypy ./src`, a wheel build and installed-wheel import/CLI smoke check, and `git diff --check`. Report preexisting failures explicitly; do not change assertions to manufacture a green suite.
- [ ] Run real tiny-Qwen multimodal forward/backward and save/reload checks offline. Fixtures and mocks alone are not evidence of backbone integration.
- [ ] Prepare real-model qualification commands and manifest validation. Where local hardware/assets permit, run the pretrained-backbone smoke check and a labeled visual overfit check, followed by document-separated calibration and evaluation. Paid/external capacity remains separately authorized.
- [ ] Record the effective image-count/resolution limits and actual memory/latency on the tested hardware. A checkpoint is visually qualified only with held-out multi-image evidence and calibration; a runtime implementation or synthetic overfit is insufficient.
- [ ] Review the complete diff against the spec, preserving legacy behavior and unrelated files. Report code-check completion separately from outstanding real-data/GPU qualification. No automatic merge, publication, or production deployment.

## Plan Self-Review

All spec sections map to Tasks 1-8: transport and upstream boundary (1/4), preparation and positions (2), backbone and revision (3), uncached serving/accounting (4), data/training/splits/calibration (5/7), export/capabilities (4/6), and validation/capacity (8). Later cache optimization and upstream conversion implementation remain explicitly outside the selected scope. All five Review Focus conditions have tests assigned to their owning tasks.

## Execution Notes

Plan written after inspecting the current checkout on `main`. Existing changes are README.md, README-UV.md, the design documentation, and start-codex.sh. Runtime source code has not been modified for this feature. Full model training requires a labeled multi-image dataset and a target hardware profile; neither is inferred from the existing text-only v19 corpus.
