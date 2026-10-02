"""Normalized page inputs must preserve identity and reject incompatible payloads."""

import base64
import io

import pytest

pytest.importorskip("PIL")

from PIL import Image
from pydantic import ValidationError

from strands_decider import schema


def image_input(identifier="page-1", *, fmt="PNG", mode="RGB", **metadata):
    data = io.BytesIO()
    Image.new(mode, (12, 8), "white").save(data, format=fmt)
    return schema.ImageInput(
        id=identifier,
        mime_type="image/jpeg" if fmt == "JPEG" else "image/png",
        data_base64=base64.b64encode(data.getvalue()).decode(),
        **metadata,
    )


def question():
    return schema.NoulQuestion(instructions="Is this an invoice?")


def test_image_only_request_accepts_empty_state():
    request = schema.SystemOneRequest(state="", images=[image_input()], questions={"q": question()})
    assert request.images[0].id == "page-1"


def test_empty_request_still_fails():
    with pytest.raises(ValidationError):
        schema.SystemOneRequest(state="", questions={"q": question()})


def test_duplicate_ids_fail():
    with pytest.raises(ValidationError, match="unique"):
        schema.SystemOneRequest(
            state="", images=[image_input(), image_input()], questions={"q": question()}
        )


def test_duplicate_pixels_preserve_order():
    from strands_decider.vision import ImageLimits, decode_images

    images = decode_images(
        [image_input("second", page_number=9), image_input("first", page_number=1)], ImageLimits()
    )
    assert [image.metadata.id for image in images] == ["second", "first"]
    assert [image.metadata.page_number for image in images] == [9, 1]
    assert images[0].pixels.size == (12, 8)


@pytest.mark.parametrize("metadata", [{"identifier": " "}, {"page_number": 0}, {"page_number": -1}])
def test_invalid_image_metadata_fails(metadata):
    with pytest.raises(ValidationError):
        image_input(**metadata)


@pytest.mark.parametrize("fmt,mode", [("TIFF", "RGB"), ("PNG", "L"), ("PNG", "RGBA")])
def test_decoding_rejects_unconverted_images(fmt, mode):
    from strands_decider.vision import ImageLimits, decode_images

    with pytest.raises(ValueError, match=r"converted|RGB"):
        decode_images([image_input(fmt=fmt, mode=mode)], ImageLimits())


def test_malformed_base64_fails():
    from strands_decider.vision import ImageLimits, decode_images

    image = image_input().model_copy(update={"data_base64": "not-base64!"})
    with pytest.raises(ValueError, match="base64"):
        decode_images([image], ImageLimits())


def test_actual_image_format_must_match_mime():
    from strands_decider.vision import ImageLimits, decode_images

    image = image_input(fmt="JPEG").model_copy(update={"mime_type": "image/png"})
    with pytest.raises(ValueError, match="MIME"):
        decode_images([image], ImageLimits())


def test_payload_and_pixels_limits_fail_without_partial_results():
    from strands_decider.vision import ImageLimits, ImagePayloadTooLarge, decode_images

    with pytest.raises(ImagePayloadTooLarge):
        decode_images([image_input()], ImageLimits(max_payload_bytes=8))
    with pytest.raises(ImagePayloadTooLarge):
        decode_images([image_input()], ImageLimits(max_pixels_per_image=95))
    with pytest.raises(ImagePayloadTooLarge):
        decode_images([image_input("a"), image_input("b")], ImageLimits(max_total_pixels=191))


def test_png_16_bit_rgb_is_not_silently_downconverted():
    import struct
    import zlib

    raw = base64.b64decode(image_input().data_base64)
    header = bytearray(raw[16:29])
    header[8] = 16
    modified = raw[:16] + bytes(header) + struct.pack(">I", zlib.crc32(b"IHDR" + header)) + raw[33:]
    image = image_input()
    image.data_base64 = base64.b64encode(modified).decode()
    from strands_decider.vision import ImageLimits, decode_images

    with pytest.raises(ValueError, match="8-bit"):
        decode_images([image], ImageLimits())
