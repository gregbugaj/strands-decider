import pytest
from fastapi.testclient import TestClient
from test_vision_inputs import image_input
from typer.testing import CliRunner
from vision_helpers import tiny_model

from strands_decider.cli import app
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest


def test_joint_images_and_multiple_questions_use_uncached_model(monkeypatch):
    model = tiny_model()
    engine = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=True))
    calls = []
    original = model.encode

    def encode(*args, **kwargs):
        calls.append(kwargs.get("image_grid_thw"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "encode", encode)
    response = engine.evaluate(
        SystemOneRequest(
            state="",
            images=[image_input("p9", page_number=9), image_input("p1", page_number=1)],
            questions={
                "type": ChoiceQuestion(
                    instructions="Classify both pages", criteria={"invoice": "", "contract": ""}
                ),
                "urgent": NoulQuestion(instructions="Is it urgent?"),
            },
        )
    )
    assert set(response.answers) == {"type", "urgent"}
    assert sum(response.answers["type"].probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert "confidence" not in response.answers["urgent"].model_dump()
    assert response.usage.input_tokens > 0 and response.usage.output_tokens == 2
    assert engine.cfg.use_prefix_cache is True
    assert len(calls) == 2 and all(grid is not None and len(grid) == 2 for grid in calls)


def test_legacy_checkpoint_rejects_images_before_attempting_text_fallback():
    model = tiny_model()
    model.config.input_mode = "text"
    engine = SystemOneEngine(model, EngineConfig(device="cpu"))
    with pytest.raises(ValueError, match=r"text.*checkpoint|multimodal"):
        engine.ask("", {"q": NoulQuestion(instructions="Yes?")}, images=[image_input()])


def test_http_image_requests_and_limits(monkeypatch):
    from strands_decider import server
    from strands_decider.vision import ImageLimits

    monkeypatch.setattr(server.StrandsDeciderModel, "load", lambda *a, **kw: tiny_model())
    client = TestClient(server.create_app("local", device="cpu"))
    request = dict(
        state="",
        images=[image_input().model_dump()],
        questions={"q": {"type": "noul", "instructions": "Invoice?"}},
    )
    assert client.post("/v1/systemone", json=request).status_code == 200
    assert client.get("/health").json()["input_mode"] == "multimodal"
    request["images"][0] = image_input(fmt="TIFF").model_dump()
    assert client.post("/v1/systemone", json=request).status_code == 422
    limited = TestClient(
        server.create_app("local", device="cpu", image_limits=ImageLimits(max_payload_bytes=8))
    )
    request["images"][0] = image_input().model_dump()
    assert limited.post("/v1/systemone", json=request).status_code == 413


def test_cli_accepts_ordered_images_without_state(tmp_path, monkeypatch):
    from PIL import Image

    import strands_decider.infer as inference

    model = tiny_model()
    engine = SystemOneEngine(model, EngineConfig(device="cpu"))
    monkeypatch.setattr(inference, "load_engine", lambda *a, **kw: engine)
    paths = [tmp_path / "page-9.png", tmp_path / "page-1.png"]
    for path in paths:
        Image.new("RGB", (12, 8), "white").save(path)
    result = CliRunner().invoke(
        app,
        ["ask", "local", "--image", str(paths[0]), "--image", str(paths[1]), "--noul", "Invoice?"],
    )
    assert result.exit_code == 0, result.output
    assert "noul" in result.output


def test_multimodal_checkpoint_accepts_no_images(monkeypatch):
    model = tiny_model()
    engine = SystemOneEngine(model, EngineConfig(device="cpu"))
    seen = []
    original = model.encode

    def encode(*args, **kwargs):
        seen.append(kwargs.get("pixel_values"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model, "encode", encode)
    response = engine.ask("Ordinary text request", {"q": NoulQuestion(instructions="Yes?")})
    assert response.answers["q"].type == "noul"
    assert seen == [None]


def test_http_body_limit_applies_before_json_parsing(monkeypatch):
    from strands_decider import server
    from strands_decider.vision import ImageLimits

    monkeypatch.setattr(server.StrandsDeciderModel, "load", lambda *a, **kw: tiny_model())
    client = TestClient(
        server.create_app("local", device="cpu", image_limits=ImageLimits(max_request_bytes=32))
    )
    assert client.post("/v1/systemone", content="x" * 33).status_code == 413


def test_cli_missing_state_fails_without_loading_checkpoint(monkeypatch):
    import strands_decider.infer as inference

    def unexpected_load(*args, **kwargs):
        raise RuntimeError("checkpoint should not load")

    monkeypatch.setattr(inference, "load_engine", unexpected_load)
    result = CliRunner().invoke(app, ["ask", "local", "--noul", "Yes?"])
    assert result.exit_code == 2 and "--state" in result.output
