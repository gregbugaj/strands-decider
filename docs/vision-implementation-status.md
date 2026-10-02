# Vision implementation verification — 2026-10-02

Optional image requests are implemented. A multimodal checkpoint accepts text-only,
one-image, and ordered multi-image requests; legacy text checkpoints reject images.
See [usage and training](vision.md). No trained visual decision checkpoint was produced.

Verified in the local Python 3.12 environment (PyTorch 2.14.1+cu130,
Transformers 5.18.0, PEFT 0.21.2):

- Full offline CPU suite after review fixes and the runnable query example:
  **282 passed, 21 skipped, 13 deselected** in the original editable checkout.
  Skipped/de-selected checks include existing GPU/external-asset and distributed gates.
- Real tiny Qwen processor/backbone forward, supervised backward, frozen visual tower,
  decoder adapter/head gradients, mixed 0/1/3-image training batches, a complete training
  step, optional-image inference, and checkpoint prediction roundtrip.
- Fresh read-only review. Fixed all four Important findings with failing regression
  tests followed by a passing full suite: replay/validation document overlap,
  unrelated-page substitution at maximum image count, incomplete processor inventories,
  and missing explicit sequence budgets.
- Ruff lint/changed-file formatting, Python 3.10 syntax parsing, documentation JSON/shell
  syntax and links, wheel build, and installed-wheel import/CLI checks.
- The [exact README image-query command](../README.md#query-an-image-now-with-pretrained-qwen)
  passed on the real cached Qwen3.5-4B weights on CPU, reading `INV-1042` and `$125.00`
  from the included synthetic RGB invoice. This uses the pretrained language head,
  independent of Decider. Three additional checks exercise real tiny-model generation
  with zero/two images and TIFF rejection. Follow-up review found no blocking issues.
- Mypy reports **the same 28 preexisting errors** on the untouched baseline and feature
  in this environment. Type checking is not green; this feature adds no new errors.

The cached pretrained **Qwen/Qwen3.5-4B** at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` passed CPU inference with 0/1/4 images
and a one-image supervised backward pass with decoder LoRA/head gradients and frozen
vision weights. [Raw smoke evidence](vision-pretrained-smoke.json) records the actual
budgets and timings. The head was untrained and pixels synthetic; this verifies runtime
integration, not classification quality. Its 16-visual-token-per-image smoke profile
differs from the larger experimental training configuration.

The 24-GiB RTX 4090 had about 6.7 GiB free at qualification time. Other workloads were
left running. Production GPU memory/latency and real-document accuracy, visual dependence,
calibration, and confidence thresholds remain unqualified. Supply independently grouped
real training, validation, calibration, and evaluation manifests for those gates.

Scope decisions:

- Implementation was developed in an isolated feature worktree; reviewed file changes
  were applied to the original working tree without commits, merges, pushes or publishing.
  The feature worktree remains available for inspection. The follow-up Qwen query example
  and README command were added directly to the original `main` working tree. The user
  will commit the integrated changes.
- Multimodal training currently supports one process/GPU, with decoder LoRA and a frozen
  vision encoder. Precomputed legacy KL and legacy head initialization are rejected;
  vision prefix caching is deferred. The cost is deferred throughput/legacy initialization
  options, rather than incorrect flattened-image slicing.
- Real tiny processors require the optional vision dependencies, including torchvision.
  CI installs matching CPU torch/torchvision wheels to exercise those tests.
- The evaluation report explicitly marks training independence unverified if train and
  validation manifests are omitted. Real-world quality remains a separate gate.
- Existing installation documentation/start scripts and baseline type-check failures
  were preserved. Only feature files and narrow documentation links were delivered.

Deferred minor from review: the ASGI body limiter extends its buffer before checking
whether an incoming chunk exceeds the limit. It still rejects over-limit requests
before JSON parsing; checking before copying would reduce transient memory for unusually
large ASGI chunks. Normal Uvicorn chunking bounds that practical overhead.

Deferred minor from the query-example review: its tiny generation tests assert successful
execution, but do not explicitly assert output-token counts or image order at the processor.
The real pretrained command verifies the documented sample answer; broader answer quality
and CUDA capacity remain unqualified.
