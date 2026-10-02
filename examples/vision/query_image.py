"""Ask the pretrained Qwen backbone a free-form question, without a Decider head.

Uses an existing local checkpoint, normalized PNG/JPEG pages, and no Hub downloads.
See the repository README for an exact command and the distinction from Decider.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

from strands_decider.vision import ImageLimits, decode_images, images_from_paths, move_model_inputs


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Qwen3.5 checkpoint directory")
    parser.add_argument("--question", required=True, help="Free-form question about the pages")
    parser.add_argument(
        "--image", action="append", default=[], help="RGB PNG/JPEG; repeat in order"
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=positive_int, default=8, help="CPU threads")
    parser.add_argument("--max-new-tokens", type=positive_int, default=128)
    parser.add_argument("--max-visual-tokens-per-image", type=positive_int, default=1024)
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("--question must contain text")
    if len(args.image) > 4:
        parser.error("at most four images are supported by this example")
    if not Path(args.model).is_dir():
        parser.error("--model must point to an existing local checkpoint directory")
    try:
        limits = ImageLimits()
        pages = decode_images(images_from_paths(args.image, limits), limits)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    image_processor = processor.image_processor
    merged_patch = image_processor.patch_size * image_processor.merge_size
    max_pixels = args.max_visual_tokens_per_image * merged_patch**2
    min_pixels = min(int(image_processor.size["shortest_edge"]), max_pixels)
    content = [{"type": "image", "image": page.pixels} for page in pages]
    content.append({"type": "text", "text": args.question})
    prompt = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(
        text=[prompt],
        images=[page.pixels for page in pages] or None,
        add_special_tokens=False,
        return_tensors="pt",
        images_kwargs={"size": {"shortest_edge": min_pixels, "longest_edge": max_pixels}},
    )
    prompt_length = inputs["input_ids"].shape[1]
    if prompt_length + args.max_new_tokens > 8192:
        parser.error("prompt plus output budget exceeds the example's 8192-token limit")
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    model = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            args.model,
            local_files_only=True,
            dtype=dtype,
            attn_implementation="eager",
        )
        .to(args.device)
        .eval()
    )
    with torch.inference_mode():
        output = model.generate(
            **move_model_inputs(dict(inputs), args.device, model),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    print(processor.batch_decode(output[:, prompt_length:], skip_special_tokens=True)[0].strip())


if __name__ == "__main__":
    main()
