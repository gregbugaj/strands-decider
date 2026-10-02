"""Exercise the runnable Qwen example with actual offline generation weights."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("torchvision")

from PIL import Image
from transformers import Qwen3_5ForConditionalGeneration
from vision_helpers import backbone, processor

SCRIPT = Path(__file__).resolve().parents[1] / "examples/vision/query_image.py"


@pytest.mark.parametrize("image_count", [0, 2])
def test_offline_query_generates_with_optional_ordered_images(tmp_path, image_count):
    proc = processor()
    proc.chat_template = (
        "{% for message in messages %}{{ message['role'] }}: "
        "{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>"
        "{% else %}{{ item['text'] }}{% endif %}{% endfor %}\n{% endfor %}assistant: "
    )
    checkpoint = tmp_path / "checkpoint"
    proc.save_pretrained(checkpoint)
    Qwen3_5ForConditionalGeneration(backbone(proc).config).save_pretrained(checkpoint)
    command = [
        sys.executable,
        str(SCRIPT),
        "--model",
        str(checkpoint),
        "--question",
        "Describe these pages.",
        "--device",
        "cpu",
        "--threads",
        "1",
        "--max-new-tokens",
        "2",
        "--max-visual-tokens-per-image",
        "16",
    ]
    for index in range(image_count):
        page = tmp_path / f"page-{index}.png"
        Image.new("RGB", (8, 8), ["red", "blue"][index]).save(page)
        command.extend(["--image", str(page)])
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode == 0, result.stderr


def test_query_rejects_tiff_before_loading_weights(tmp_path):
    page = tmp_path / "page.tif"
    Image.new("RGB", (8, 8)).save(page)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--model",
            str(tmp_path),
            "--image",
            str(page),
            "--question",
            "Describe the page.",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "convert upstream" in result.stderr
