import pytest
from test_vision_training import records
from vision_helpers import tiny_model

from strands_decider.evaluate import Prediction
from strands_decider.infer import EngineConfig, SystemOneEngine


def test_visual_report_keeps_controls_and_failures_separate(tmp_path):
    from vision_eval import evaluate_conditions

    rows = records(tmp_path)

    def predict(ex):
        if len(ex.images) == 3:
            raise ValueError("example budget exceeded")
        probs = [0.1, 0.9]
        return Prediction(ex.task, ex.kind, 2, ex.label, probs), {
            "preprocessing_ms": 1.0,
            "latency_ms": 2.0,
            "input_tokens": 64,
            "peak_gpu_bytes": None,
        }

    report = evaluate_conditions(None, rows, predictor=predict)
    combined = report["conditions"]["combined"]
    assert combined["attempted"] == 3 and combined["failed"] == 1
    assert combined["metrics"]["accuracy"] == 1
    assert combined["metrics"]["brier"] == pytest.approx(0.02)
    assert "removed_page" in report["controls"] and "unrelated_page" in report["controls"]
    assert report["conditions"]["image_only"]["skipped"] == 1
    assert combined["by_image_count"]["0"]["accuracy"] == 1


def test_visual_report_uses_real_optional_image_path(tmp_path):
    from vision_eval import evaluate_conditions

    engine = SystemOneEngine(tiny_model(), EngineConfig(device="cpu"))
    rows = records(tmp_path)
    report = evaluate_conditions(engine, rows)
    assert report["conditions"]["combined"]["succeeded"] == 3
    assert report["conditions"]["ocr_only"]["succeeded"] == 3
    assert report["conditions"]["combined"]["resource"]["input_tokens"] > 0
    assert report["config"]["input_mode"] == "multimodal"


def test_calibration_and_evaluation_groups_must_be_independent(tmp_path):
    from vision_eval import validate_manifests

    rows = records(tmp_path)
    with pytest.raises(ValueError, match="overlap"):
        validate_manifests(calibration=[rows[1]], evaluation=[rows[2]])


def test_unrelated_page_substitutes_at_maximum_count(tmp_path):
    import copy
    import hashlib

    from PIL import Image
    from vision_eval import condition_example, evaluate_conditions

    from strands_decider.data.format import ImageAsset

    rows = records(tmp_path)
    original = copy.copy(rows[2])
    original.images = [copy.copy(image) for image in rows[2].images]
    duplicate = copy.copy(original.images[0])
    duplicate.id = "fourth-image"
    original.images.append(duplicate)
    donor = copy.copy(rows[1])
    file = tmp_path / "donor.png"
    Image.new("RGB", (8, 8), "black").save(file)
    asset = ImageAsset(
        "donor",
        "donor.png",
        hashlib.sha256(file.read_bytes()).hexdigest(),
        source_id="independent-source",
    )
    asset.resolve(tmp_path)
    donor.images = [asset]
    donor.document_id = "independent-document"
    changed = condition_example(original, "unrelated_page", donor)
    assert len(changed.images) == 4
    assert changed.images[:3] == original.images[:3]
    assert changed.images[-1].source_id == "independent-source"
    engine = SystemOneEngine(tiny_model(), EngineConfig(device="cpu"))
    report = evaluate_conditions(engine, [original, donor])
    assert report["controls"]["unrelated_page"]["failed"] == 0
    assert report["controls"]["unrelated_page"]["succeeded"] == 2
