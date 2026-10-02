"""Normalized page transport and shared Qwen multimodal preparation."""

from __future__ import annotations

import base64
import binascii
import io
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Any, Literal, cast

import torch
from torch.nn.utils.rnn import pad_sequence

from .prompting import RenderedQuestion, render_content
from .schema import Content, ImageInput


@dataclass(frozen=True)
class ImageLimits:
    """CPU transport limits, independent of the checkpoint's visual token budgets."""

    max_request_bytes: int = 40 * 1024 * 1024
    max_payload_bytes: int = 32 * 1024 * 1024
    max_decoded_bytes: int = 24 * 1024 * 1024
    max_pixels_per_image: int = 16_000_000
    max_total_pixels: int = 32_000_000

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("image transport limits must be positive")


class ImagePayloadTooLarge(ValueError):
    """Transport or decoded image size exceeds the server's configured limit."""


@dataclass
class DecodedImage:
    pixels: Any
    metadata: ImageInput


def decode_images(images: Sequence[ImageInput], limits: ImageLimits) -> list[DecodedImage]:
    """Validate all pages atomically; preserve order and independent pixel buffers."""
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise ValueError('Image inputs require the "vision" extra: install .[vision]') from exc

    encoded_size = sum(len(image.data_base64) for image in images)
    if encoded_size > limits.max_payload_bytes:
        raise ImagePayloadTooLarge(
            f"image payload bytes {encoded_size} exceed {limits.max_payload_bytes}"
        )
    decoded_bytes = total_pixels = 0
    results: list[DecodedImage] = []
    for image in images:
        try:
            raw = base64.b64decode(image.data_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"image {image.id!r} contains invalid base64") from exc
        decoded_bytes += len(raw)
        if decoded_bytes > limits.max_decoded_bytes:
            raise ImagePayloadTooLarge(
                f"decoded image bytes {decoded_bytes} exceed {limits.max_decoded_bytes}"
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw), formats=["PNG", "JPEG"]) as pixels:
                    expected = "PNG" if image.mime_type == "image/png" else "JPEG"
                    if pixels.format != expected:
                        raise ValueError(f"image {image.id!r} decoded format does not match MIME")
                    if pixels.mode != "RGB" or getattr(pixels, "n_frames", 1) != 1:
                        raise ValueError(f"image {image.id!r} must be a converted single RGB page")
                    count = pixels.width * pixels.height
                    total_pixels += count
                    if (
                        count > limits.max_pixels_per_image
                        or total_pixels > limits.max_total_pixels
                    ):
                        raise ImagePayloadTooLarge(
                            f"image pixels {count} (total {total_pixels}) exceed limits "
                            f"{limits.max_pixels_per_image} per page / {limits.max_total_pixels} total"
                        )
                    if pixels.format == "PNG" and raw[24] != 8:
                        raise ValueError(
                            f"image {image.id!r} must be converted to 8-bit RGB upstream"
                        )
                    if pixels.getexif().get(274, 1) != 1:
                        raise ValueError(
                            f"image {image.id!r} must have orientation normalized upstream"
                        )
                    pixels.load()
                    results.append(DecodedImage(pixels.copy(), image))
        except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
            raise ImagePayloadTooLarge(f"image {image.id!r} exceeds decoded pixel limits") from exc
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(
                f"image {image.id!r} must be a valid converted RGB PNG/JPEG page; "
                "convert TIFF sources upstream"
            ) from exc
    return results


@dataclass(frozen=True)
class VisionBudget:
    max_images: int
    max_visual_tokens_per_image: int
    max_total_visual_tokens: int
    max_length: int

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("visual token and image budgets must be positive")


@dataclass
class PreparedVisionBatch:
    tensors: dict[str, torch.Tensor]
    opt_idx: torch.Tensor
    answer_positions: torch.Tensor
    visual_tokens: list[int]
    sequence_tokens: list[int]
    images: list[list[DecodedImage]]


def _expanded_indices(
    original: Sequence[int], final: Sequence[int], image_id: int, counts: Sequence[int]
) -> list[int]:
    """Prove alignment with the processor; only image placeholders may expand."""
    mapping: list[int] = []
    cursor = image = 0
    for token in original:
        n = 1
        if token == image_id:
            if image >= len(counts):
                raise ValueError("unexpected image placeholder in text")
            n = counts[image]
            image += 1
        if n < 1 or list(final[cursor : cursor + n]) != [token] * n:
            raise ValueError("processor token alignment changed; cannot locate option positions")
        mapping.append(cursor)
        cursor += n
    if cursor != len(final) or image != len(counts):
        raise ValueError("processor image/token counts do not match the supplied pages")
    return mapping


def prepare_vision_batch(
    processor: Any,
    states: Sequence[Content],
    images: Sequence[Sequence[DecodedImage]],
    questions: Sequence[RenderedQuestion],
    budget: VisionBudget,
) -> PreparedVisionBatch:
    """Use one deterministic processor path for serving and supervised batches."""
    if not states or not (len(states) == len(images) == len(questions)):
        raise ValueError("states, image groups, and questions must have equal nonzero lengths")
    tokenizer = processor.tokenizer
    image_id = int(processor.image_token_id)
    merge = int(processor.image_processor.merge_size)
    patch = int(processor.image_processor.patch_size)
    max_pixels = budget.max_visual_tokens_per_image * (patch * merge) ** 2
    min_pixels = min(int(processor.image_processor.size["shortest_edge"]), max_pixels)
    reserved = (
        processor.image_token,
        processor.video_token,
        processor.vision_start_token,
        processor.vision_end_token,
    )
    encoded_rows: list[dict[str, torch.Tensor]] = []
    options: list[list[int]] = []
    answers: list[int] = []
    visual_totals: list[int] = []
    lengths: list[int] = []
    for state, group, rq in zip(states, images, questions, strict=True):
        if len(group) > budget.max_images:
            raise ValueError(f"image count {len(group)} exceeds {budget.max_images}")
        if any(token in rq.text for token in reserved):
            raise ValueError("question text must not contain reserved multimodal tokens")
        prefix = f"<state>\n{escape(render_content(state))}\n"
        for page in group:
            meta = page.metadata
            prefix += f'<image id="{escape(meta.id, quote=True)}"'
            if meta.source_id is not None:
                prefix += f' source="{escape(meta.source_id, quote=True)}"'
            if meta.page_number is not None:
                prefix += f' page="{meta.page_number}"'
            prefix += ">\n" + processor.vision_start_token + processor.image_token
            prefix += processor.vision_end_token + "\n"
            if meta.text:
                prefix += escape(meta.text) + "\n"
            prefix += "</image>\n"
        prefix += "</state>\n"
        prompt = prefix + rq.text
        original = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        encoded = processor(
            text=[prompt],
            images=[page.pixels for page in group] or None,
            return_tensors="pt",
            add_special_tokens=False,
            padding=False,
            images_kwargs={"size": {"shortest_edge": min_pixels, "longest_edge": max_pixels}},
        )
        ids = encoded["input_ids"][0].tolist()
        grids = encoded.get("image_grid_thw")
        counts = [] if grids is None else (grids.prod(dim=-1) // merge**2).tolist()
        if len(counts) != len(group):
            raise ValueError("processor image grids do not match supplied page count")
        if any(count > budget.max_visual_tokens_per_image for count in counts):
            raise ValueError(
                f"per-image visual tokens {counts} exceed {budget.max_visual_tokens_per_image}"
            )
        total = sum(counts)
        if total > budget.max_total_visual_tokens:
            raise ValueError(f"visual tokens {total} exceed {budget.max_total_visual_tokens}")
        if len(ids) > budget.max_length:
            raise ValueError(
                f"sequence tokens {len(ids)} exceed {budget.max_length}; select fewer pages or text"
            )
        mapping = _expanded_indices(original["input_ids"], ids, image_id, counts)
        row: list[int] = []
        for start, end in rq.option_spans:
            a, b = len(prefix) + start, len(prefix) + end
            positions = [
                i
                for i, (lo, hi) in enumerate(original["offset_mapping"])
                if hi > lo and lo >= a and hi <= b
            ]
            if not positions:
                raise ValueError("option has no surviving tokens after image processing")
            row.append(mapping[positions[-1]])
        if not row or not rq.text.endswith("<answer>"):
            raise ValueError("vision questions require options and a final answer marker")
        options.append(row)
        answers.append(mapping[-1])
        visual_totals.append(total)
        lengths.append(len(ids))
        encoded_rows.append(
            {
                key: value
                for key, value in encoded.items()
                if key
                in {
                    "input_ids",
                    "attention_mask",
                    "mm_token_type_ids",
                    "pixel_values",
                    "image_grid_thw",
                }
            }
        )
    tensors: dict[str, torch.Tensor] = {}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
        if all(key in row for row in encoded_rows):
            pad = (tokenizer.pad_token_id or 0) if key == "input_ids" else 0
            tensors[key] = pad_sequence(
                [row[key][0] for row in encoded_rows], batch_first=True, padding_value=pad
            )
    for key in ("pixel_values", "image_grid_thw"):
        values = [row[key] for row in encoded_rows if key in row]
        if values:
            tensors[key] = torch.cat(values, dim=0)
    width = max(len(row) for row in options)
    return PreparedVisionBatch(
        tensors=tensors,
        opt_idx=torch.tensor([row + [-1] * (width - len(row)) for row in options]),
        answer_positions=torch.tensor(answers),
        visual_tokens=visual_totals,
        sequence_tokens=lengths,
        images=[list(group) for group in images],
    )


def move_model_inputs(
    tensors: dict[str, torch.Tensor], device: Any, torso: Any
) -> dict[str, torch.Tensor]:
    """Move model tensors without converting integer grids/positions into floats."""
    parameter = next(torso.parameters()) if "pixel_values" in tensors else None
    return {
        key: value.to(
            device=device,
            dtype=parameter.dtype
            if parameter is not None and key == "pixel_values"
            else value.dtype,
        )
        for key, value in tensors.items()
    }


def images_from_paths(paths: Sequence[str], limits: ImageLimits | None = None) -> list[ImageInput]:
    """Read ordered, already normalized CLI images; decoding checks actual format."""
    from pathlib import Path

    limits = limits or ImageLimits()
    images = []
    for index, path in enumerate(paths, 1):
        file = Path(path)
        if file.stat().st_size > limits.max_decoded_bytes:
            raise ImagePayloadTooLarge(
                f"{file}: file bytes exceed allowed {limits.max_decoded_bytes}"
            )
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
            file.suffix.lower()
        )
        if mime is None:
            raise ValueError(f"{file}: convert upstream to an 8-bit RGB PNG/JPEG before input")
        images.append(
            ImageInput(
                id=f"image-{index}",
                mime_type=cast(Literal["image/png", "image/jpeg"], mime),
                data_base64=base64.b64encode(file.read_bytes()).decode("ascii"),
            )
        )
    decode_images(images, limits)
    return images
