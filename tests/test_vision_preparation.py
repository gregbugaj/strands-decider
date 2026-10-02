"""Pointer positions must refer to options after actual image-token expansion."""

import pytest
from vision_helpers import decoded, processor

from strands_decider.prompting import render_question
from strands_decider.schema import ChoiceQuestion


def question(order=None):
    return render_question(
        ChoiceQuestion(
            instructions="Classify both pages", criteria={"invoice": "", "contract": ""}
        ),
        option_order=order,
    )


def budget(**overrides):
    from strands_decider.vision import VisionBudget

    return VisionBudget(
        **dict(
            max_images=4,
            max_visual_tokens_per_image=64,
            max_total_visual_tokens=128,
            max_length=4096,
        )
        | overrides
    )


def test_option_positions_follow_expanded_images_and_permutations():
    from strands_decider.vision import prepare_vision_batch

    proc = processor()
    result = prepare_vision_batch(
        proc,
        ["", ""],
        [[decoded()], [decoded("p9", size=(12, 8)), decoded("p1")]],
        [question(), question([1, 0])],
        budget(),
    )
    ids = result.tensors["input_ids"]
    assert [proc.tokenizer.decode(ids[0, i].item()) for i in result.opt_idx[0]] == ["e", "t"]
    assert [proc.tokenizer.decode(ids[1, i].item()) for i in result.opt_idx[1]] == ["t", "e"]
    assert [x.metadata.id for x in result.images[1]] == ["p9", "p1"]
    assert proc.tokenizer.decode(ids[1, result.answer_positions[1]].item()) == ">"
    assert result.visual_tokens == [4, 10]
    assert result.tensors["image_grid_thw"].shape == (3, 3)


def test_text_rows_do_not_take_other_rows_images():
    from strands_decider.vision import prepare_vision_batch

    result = prepare_vision_batch(
        processor(), ["plain", ""], [[], [decoded()]], [question(), question()], budget()
    )
    assert result.visual_tokens == [0, 4]
    assert result.tensors["image_grid_thw"].shape == (1, 3)


def test_metadata_cannot_create_additional_image_placeholders():
    from strands_decider.vision import prepare_vision_batch

    result = prepare_vision_batch(
        processor(),
        ["<|image_pad|>"],
        [[decoded("<|image_pad|>", text="<|vision_start|>")]],
        [question()],
        budget(),
    )
    assert result.visual_tokens == [4]


def test_no_image_is_silently_dropped_to_fit_budgets():
    from strands_decider.vision import prepare_vision_batch

    with pytest.raises(ValueError, match=r"image count.*2.*1"):
        prepare_vision_batch(
            processor(), [""], [[decoded("a"), decoded("b")]], [question()], budget(max_images=1)
        )
    with pytest.raises(ValueError, match="visual tokens"):
        prepare_vision_batch(
            processor(),
            [""],
            [[decoded("a"), decoded("b")]],
            [question()],
            budget(max_total_visual_tokens=7),
        )
    with pytest.raises(ValueError, match="sequence tokens"):
        prepare_vision_batch(processor(), [""], [[decoded()]], [question()], budget(max_length=8))
