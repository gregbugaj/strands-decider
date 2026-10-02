import pytest
import torch
from vision_helpers import decoded, model_config, tiny_model

from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
from strands_decider.prompting import render_question
from strands_decider.schema import ChoiceQuestion
from strands_decider.vision import prepare_vision_batch


def test_visual_config_requires_pinned_revision_and_budgets():
    with pytest.raises(ValueError, match="revision"):
        model_config(base_revision=None)
    with pytest.raises(ValueError, match=r"budget|positive"):
        model_config(max_images=None)
    assert StrandsDeciderConfig().input_mode == "text"


def prepared(model):
    rq = render_question(
        ChoiceQuestion(instructions="Classify pages", criteria={"invoice": "", "contract": ""})
    )
    return prepare_vision_batch(
        model.processor, [""], [[decoded("a"), decoded("b")]], [rq], model.config.vision_budget()
    )


def test_real_multimodal_forward_and_decoder_gradients():
    model = tiny_model(use_lora=True, lora_targets=["q_proj", "v_proj"])
    model.attach_lora()
    batch = prepared(model)
    out = model(
        **batch.tensors,
        opt_idx=batch.opt_idx,
        answer_positions=batch.answer_positions,
        n_slots=torch.tensor([2]),
        labels=torch.tensor([1]),
    )
    assert out["logits"].shape == (1, 2)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert model.head.q.weight.grad.abs().sum() > 0
    adapters = [p for name, p in model.torso.named_parameters() if "lora_B" in name]
    assert adapters and any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapters)
    vision = [p for name, p in model.torso.named_parameters() if ".visual." in name]
    assert vision and all(not p.requires_grad and p.grad is None for p in vision)


def test_visual_checkpoint_roundtrip_requires_processor(tmp_path):
    model = tiny_model(use_lora=True, lora_targets=["q_proj", "v_proj"])
    base = tmp_path / "base"
    model.torso.save_pretrained(base)
    model.processor.save_pretrained(base)
    model.config.base_model = str(base)
    model.attach_lora()
    model.eval()
    batch = prepared(model)
    kwargs = dict(
        batch.tensors,
        opt_idx=batch.opt_idx,
        answer_positions=batch.answer_positions,
        n_slots=torch.tensor([2]),
    )
    before = model(**kwargs)["logits"]
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(str(checkpoint))
    restored = StrandsDeciderModel.load(str(checkpoint)).eval()
    torch.testing.assert_close(restored(**kwargs)["logits"], before, atol=1e-6, rtol=1e-5)
    (checkpoint / model.config.processor_files[0]).unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="processor"):
        StrandsDeciderModel.load(str(checkpoint))


def test_text_model_rejects_pixel_tensors_directly():
    model = tiny_model()
    batch = prepared(model)
    model.config.input_mode = "text"
    with pytest.raises(ValueError, match="text checkpoint"):
        model(
            **batch.tensors,
            opt_idx=batch.opt_idx,
            answer_positions=batch.answer_positions,
            n_slots=torch.tensor([2]),
        )


def test_visual_reload_without_lora_still_freezes_encoder(tmp_path):
    model = tiny_model()
    base = tmp_path / "base"
    model.torso.save_pretrained(base)
    model.processor.save_pretrained(base)
    model.config.base_model = str(base)
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(str(checkpoint))
    loaded = StrandsDeciderModel.load(str(checkpoint))
    assert all(not p.requires_grad for p in loaded.torso.visual.parameters())


def test_full_model_text_reference_uses_text_config_tied_embeddings():
    model = tiny_model()
    model.torso.config.text_config.tie_word_embeddings = True
    model._output_embedding()


def test_serialized_visual_config_requires_explicit_sequence_budget(tmp_path):
    import json
    from dataclasses import asdict

    raw = asdict(model_config())
    raw.pop("max_length")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="max_length"):
        StrandsDeciderConfig.from_json(str(path))
    path.write_text(json.dumps({"base_model": "legacy"}))
    assert StrandsDeciderConfig.from_json(str(path)).max_length == 3072
