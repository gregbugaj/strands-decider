import json

import pytest
from vision_helpers import tiny_model

from strands_decider import hf_export
from strands_decider.modeling import CONFIG_NAME


def checkpoint(tmp_path):
    model = tiny_model(use_lora=True, lora_targets=["q_proj", "v_proj"])
    model.attach_lora()
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(str(checkpoint))
    return checkpoint, model.config


def build(checkpoint, output):
    output.mkdir()
    return hf_export.build(
        str(checkpoint), str(output), None, None, [], [], "vision-experiment", "test-run", "final"
    )


def test_visual_export_copies_processor_and_uses_actual_revision(tmp_path):
    checkpoint_path, cfg = checkpoint(tmp_path)
    out = tmp_path / "out"
    build(checkpoint_path, out)
    hf_export.verify(str(out))
    prov = json.loads((out / "provenance.json").read_text())
    assert prov["base_model_revision"] == cfg.base_revision
    assert prov["input_mode"] == "multimodal"
    assert prov["max_images"] == 4
    for file in cfg.processor_files:
        assert (out / file).read_bytes() == (checkpoint_path / file).read_bytes()
    card = (out / "README.md").read_text()
    assert "v19" not in card and "0.641" not in card and "not yet qualified" in card
    out2 = tmp_path / "out2"
    build(checkpoint_path, out2)
    assert (out / "MANIFEST.sha256").read_bytes() == (out2 / "MANIFEST.sha256").read_bytes()
    # A self-consistent general manifest must not bypass missing required processor files.
    (out / cfg.processor_files[0]).unlink()
    (out / "MANIFEST.sha256").write_text(hf_export.manifest(str(out)))
    with pytest.raises(SystemExit, match="processor"):
        hf_export.verify(str(out))


def test_visual_export_requires_pinned_revision(tmp_path):
    ckpt, _ = checkpoint(tmp_path)
    config = json.loads((ckpt / CONFIG_NAME).read_text())
    config.pop("base_revision")
    (ckpt / CONFIG_NAME).write_text(json.dumps(config))
    with pytest.raises((ValueError, SystemExit), match="revision"):
        build(ckpt, tmp_path / "out")


def test_processor_inventory_cannot_hide_missing_configuration(tmp_path):
    ckpt, _ = checkpoint(tmp_path)
    out = tmp_path / "out"
    build(ckpt, out)
    raw = json.loads((out / CONFIG_NAME).read_text())
    raw["processor_files"] = ["processor/tokenizer.json"]
    (out / CONFIG_NAME).write_text(json.dumps(raw))
    for file in (out / "processor").iterdir():
        if file.name != "tokenizer.json":
            file.unlink()
    (out / "MANIFEST.sha256").write_text(hf_export.manifest(str(out)))
    with pytest.raises(SystemExit, match="processor"):
        hf_export.verify(str(out))


def test_visual_export_rejects_undeclared_sequence_budget(tmp_path):
    ckpt, _ = checkpoint(tmp_path)
    raw = json.loads((ckpt / CONFIG_NAME).read_text())
    raw.pop("max_length")
    (ckpt / CONFIG_NAME).write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="max_length"):
        build(ckpt, tmp_path / "out")
