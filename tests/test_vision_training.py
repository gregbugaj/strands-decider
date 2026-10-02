import hashlib
import json

import pytest
import torch
from torch.utils.data import DataLoader
from vision_helpers import tiny_model

from strands_decider.data.collate import CollatorConfig, SystemOneCollator
from strands_decider.data.format import read_jsonl, split_examples, write_jsonl
from strands_decider.evaluate import collect_logits, partition_examples
from strands_decider.train import evaluate_loss


def records(tmp_path):
    from PIL import Image

    assets = []
    for i in range(3):
        path = tmp_path / f"p{i}.png"
        Image.new("RGB", (8 + 4 * i, 8), "white").save(path)
        assets.append(
            dict(
                id=f"p{i}",
                path=path.name,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_id="source-a",
                page_number=i + 1,
            )
        )
    data = [
        dict(
            kind="choice",
            state="context",
            instructions="Document type?",
            options=[["invoice", ""], ["contract", ""]],
            label=1,
            images=assets[:n],
            document_id=f"doc-{n}",
        )
        for n in (0, 1, 3)
    ]
    manifest = tmp_path / "rows.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in data))
    return list(read_jsonl(str(manifest)))


def collator(model, train=True):
    return SystemOneCollator(
        model.tokenizer,
        CollatorConfig(head_type="pointer", seed=7),
        processor=model.processor,
        vision_budget=model.config.vision_budget(),
        train=train,
    )


def test_relative_assets_hashes_and_serialization(tmp_path):
    rows = records(tmp_path)
    assert len(rows[2].to_images()) == 3
    out = tmp_path / "copy.jsonl"
    write_jsonl(str(out), rows)
    assert json.loads(out.read_text().splitlines()[2])["images"][0]["path"] == "p0.png"
    rows[2].images[0].sha256 = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        rows[2].to_images()


def test_mixed_batch_keeps_pages_and_option_labels(tmp_path):
    model = tiny_model(use_lora=True, lora_targets=["q_proj", "v_proj"])
    model.attach_lora()
    rows = records(tmp_path)
    batch = collator(model)(rows)
    assert batch["visual_rows"].tolist() == [False, True, True]
    assert len(batch["image_grid_thw"]) == 4
    # Option shuffling must still bind the stored contract label to its own line.
    for i, label in enumerate(batch["labels"].tolist()):
        pos = batch["opt_idx"][i, label]
        assert model.tokenizer.decode(batch["input_ids"][i, pos : pos + 1]) == "t"
    from strands_decider.train import model_inputs

    out = model(**model_inputs(batch))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None for p in model.head.parameters())
    assert all(not p.requires_grad for n, p in model.torso.named_parameters() if ".visual." in n)
    loader = DataLoader(rows, batch_size=3, collate_fn=collator(model, train=False))
    assert torch.isfinite(torch.tensor(evaluate_loss(model, loader, "cpu")["val_loss"]))
    logits, _, _, _ = collect_logits(model, rows, device="cpu", batch_size=3)
    assert logits.shape == (3, 2)


def test_optional_images_train_as_text_and_do_not_enter_visual_reference(tmp_path):
    from strands_decider.train import text_reference

    model = tiny_model()
    rows = records(tmp_path)
    batch = collator(model)(rows)
    seen = []

    def reference(ids, mask, slots):
        seen.append(len(ids))
        return torch.zeros(len(ids), 2), torch.ones(len(ids), dtype=torch.bool)

    model.frozen_slot_log_probs = reference
    _, eligible = text_reference(model, batch)
    assert seen == [1] and eligible.tolist() == [True, False, False]
    rows[1].teacher = [0.5, 0.5]
    with pytest.raises(ValueError, match=r"visual.*teacher"):
        collator(model)(rows)


def test_document_source_and_content_groups_cannot_leak(tmp_path):
    from strands_decider.data.format import assert_disjoint

    rows = records(tmp_path)
    rows[1].document_id = "different-id"
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint(train=[rows[1]], validation=[rows[2]])
    train, _val = split_examples(rows, val_fraction=0.5)
    assert (rows[1] in train) == (rows[2] in train)
    calib = partition_examples(rows, "calib")
    test = partition_examples(rows, "test")
    assert (rows[1] in calib) == (rows[2] in calib)
    assert_disjoint(calibration=calib, evaluation=test)
    rows[1].document_id = None
    with pytest.raises(ValueError, match="document_id"):
        split_examples(rows)


def test_one_training_step_uses_images_and_saves_processor(tmp_path, monkeypatch):
    from strands_decider.data.format import Example
    from strands_decider.modeling import StrandsDeciderModel
    from strands_decider.train import TrainConfig, train

    records(tmp_path)
    validation = tmp_path / "validation.jsonl"
    write_jsonl(
        str(validation),
        [
            Example(
                "choice",
                "validation text",
                "Type?",
                [["invoice", ""], ["contract", ""]],
                0,
                document_id="validation-document",
            )
        ],
    )
    model = tiny_model(use_lora=True, lora_targets=["q_proj", "v_proj"])
    model.attach_lora()
    monkeypatch.setattr(StrandsDeciderModel, "from_pretrained_base", lambda *a, **kw: model)
    config = TrainConfig(
        train_files=[str(tmp_path / "rows.jsonl")],
        val_files=[str(validation)],
        input_mode="multimodal",
        base_revision="0" * 40,
        prompt_format="vision-v1",
        head_type="pointer",
        max_images=4,
        max_visual_tokens_per_image=64,
        max_total_visual_tokens=128,
        max_length=4096,
        micro_batch_size=3,
        grad_accum=1,
        max_steps=1,
        gradient_checkpointing=False,
        output_dir=str(tmp_path / "trained"),
    )
    train(config)
    assert (tmp_path / "trained" / "dataset_manifest.json").is_file()
    assert model.config.processor_files
    assert all((tmp_path / "trained" / file).is_file() for file in model.config.processor_files)


def test_pointer_mixed_score_targets_allow_more_options_than_legacy_slots():
    from strands_decider.data.format import Example

    model = tiny_model()
    rows = [
        Example("choice", "Text", "Pick?", [[f"option{i}", ""] for i in range(30)], 29),
        Example("score", "Text", "Rate?", [["low", "low"], ["high", "high"]], 1),
    ]
    batch = collator(model, train=False)(rows)
    assert batch["label_dist"].shape == (2, 30)
    assert batch["label_dist"][0, 29] == 1
    from strands_decider.train import model_inputs

    assert torch.isfinite(model(**model_inputs(batch))["loss"])


def test_replay_rows_cannot_overlap_validation_documents(tmp_path, monkeypatch):
    from strands_decider.data.format import Example
    from strands_decider.modeling import StrandsDeciderModel
    from strands_decider.train import TrainConfig, train

    records(tmp_path)
    validation = tmp_path / "validation.jsonl"
    replay = tmp_path / "replay.jsonl"
    held = Example(
        "choice",
        "Held-out document",
        "Type?",
        [["a", ""], ["b", ""]],
        0,
        document_id="held-out-document",
    )
    write_jsonl(str(validation), [held])
    write_jsonl(str(replay), [held])
    model = tiny_model(use_lora=True, lora_targets=["q_proj", "v_proj"])
    model.attach_lora()
    model.torso.config.text_config.tie_word_embeddings = True
    monkeypatch.setattr(StrandsDeciderModel, "from_pretrained_base", lambda *a, **kw: model)
    cfg = TrainConfig(
        train_files=[str(tmp_path / "rows.jsonl")],
        val_files=[str(validation)],
        kl_only_files=[str(replay)],
        kl_frozen_weight=0.1,
        input_mode="multimodal",
        base_revision="0" * 40,
        prompt_format="vision-v1",
        head_type="pointer",
        max_images=4,
        max_visual_tokens_per_image=64,
        max_total_visual_tokens=128,
        max_length=4096,
        micro_batch_size=1,
        grad_accum=1,
        max_steps=1,
        gradient_checkpointing=False,
        output_dir=str(tmp_path / "trained"),
    )
    with pytest.raises(ValueError, match="overlap"):
        train(cfg)
