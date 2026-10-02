"""Small real Qwen processors and backbones without Hub downloads."""

import pytest

pytest.importorskip("PIL")
pytest.importorskip("torchvision")

from PIL import Image
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)

from strands_decider.schema import ImageInput


def processor():
    special = [
        "[PAD]",
        "[UNK]",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    vocab = {token: i for i, token in enumerate(special)}
    vocab.update(
        {
            token: i + len(special)
            for i, token in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
        }
    )
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        additional_special_tokens=special[2:],
    )
    return Qwen3VLProcessor(
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
        image_processor=Qwen2VLImageProcessor(
            patch_size=2,
            temporal_patch_size=1,
            merge_size=2,
            min_pixels=16,
            max_pixels=256,
        ),
    )


def decoded(identifier="page-1", size=(8, 8), **metadata):
    from strands_decider.vision import DecodedImage

    return DecodedImage(
        Image.new("RGB", size, "white"),
        ImageInput(id=identifier, mime_type="image/png", data_base64="unused", **metadata),
    )


def backbone(proc):
    from transformers import Qwen3_5Config, Qwen3_5Model

    return Qwen3_5Model(
        Qwen3_5Config(
            image_token_id=proc.image_token_id,
            video_token_id=proc.video_token_id,
            vision_start_token_id=proc.vision_start_token_id,
            vision_end_token_id=proc.vision_end_token_id,
            text_config={
                "vocab_size": len(proc.tokenizer),
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "layer_types": ["full_attention"],
                "partial_rotary_factor": 1.0,
                "rope_parameters": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "mrope_section": [2, 3, 3],
                },
            },
            vision_config={
                "depth": 1,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_heads": 2,
                "patch_size": 2,
                "temporal_patch_size": 1,
                "spatial_merge_size": 2,
                "out_hidden_size": 32,
                "num_position_embeddings": 64,
            },
        )
    )


def model_config(**overrides):
    from strands_decider.modeling import StrandsDeciderConfig

    values = dict(
        input_mode="multimodal",
        head_type="pointer",
        torch_dtype="float32",
        base_revision="0" * 40,
        prompt_format="vision-v1",
        max_images=4,
        max_visual_tokens_per_image=64,
        max_total_visual_tokens=128,
        max_length=4096,
        pointer_dim=8,
        use_lora=False,
    )
    return StrandsDeciderConfig(**(values | overrides))


def tiny_model(**overrides):
    from strands_decider.modeling import StrandsDeciderModel

    proc = processor()
    model = StrandsDeciderModel(model_config(**overrides), backbone(proc), proc.tokenizer)
    model.processor = proc
    return model
