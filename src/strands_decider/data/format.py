"""The on-disk training example, and JSONL round-tripping.

One example is exactly one (state, question, answer) triple -- the same unit the
serving API handles, so nothing about the input distribution shifts between train
and inference. Multi-question requests are simply several examples that happen to
share a state.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import os
import random
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any, Literal, cast

from ..schema import ChoiceQuestion, ImageInput, NoulQuestion, Question, ScoreQuestion


@dataclass
class ImageAsset:
    id: str
    path: str
    sha256: str
    source_id: str | None = None
    page_number: int | None = None
    text: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("image asset requires an id and lowercase sha256")
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise ValueError("image asset path must be relative to its manifest")
        if self.page_number is not None and self.page_number <= 0:
            raise ValueError("page_number must be positive")

    def resolve(self, manifest_directory: Path) -> None:
        self._resolved_path = manifest_directory / self.path

    def to_input(self) -> ImageInput:
        resolved = getattr(self, "_resolved_path", None)
        if resolved is None:
            raise ValueError(
                "load image assets through read_jsonl to resolve manifest-relative paths"
            )
        from ..vision import ImageLimits

        file = Path(resolved)
        if file.stat().st_size > ImageLimits().max_decoded_bytes:
            raise ValueError(f"asset {self.id!r} exceeds decoded byte limit")
        raw = file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError(f"asset {self.id!r} sha256 mismatch")
        mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
            file.suffix.lower()
        )
        if mime is None:
            raise ValueError("assets must be converted RGB PNG/JPEG pages")
        return ImageInput(
            id=self.id,
            mime_type=cast(Literal["image/png", "image/jpeg"], mime),
            data_base64=base64.b64encode(raw).decode("ascii"),
            source_id=self.source_id,
            page_number=self.page_number,
            text=self.text,
        )


@dataclass
class Example:
    """A single supervised item.

    `label` is an index into the *canonical* option order:
      - noul:   0 = false, 1 = true
      - choice: position in `options`
      - score:  the level index (which is also its position in `options`)

    Option shuffling happens at collate time, not here, so one stored example
    yields a different slot binding on every epoch.
    """

    kind: str  # "noul" | "choice" | "score"
    state: Any
    instructions: str
    # Canonical option order: [(name, description), ...]
    options: list[list[str]]
    label: int
    task: str = "unknown"  # provenance, for per-task eval breakdowns
    weight: float = 1.0
    # Alternative phrasings of the same question. Sampled per epoch at collate time,
    # exactly as option order is. Twelve of fourteen tasks originally carried a single
    # instruction string, so the model never saw a task asked two ways -- and held-out
    # accuracy on weak tasks moves by up to 0.15 on rewording alone.
    instruction_variants: list[str] = field(default_factory=list)

    images: list[ImageAsset] = field(default_factory=list)
    document_id: str | None = None

    def __post_init__(self) -> None:
        if len({image.id for image in self.images}) != len(self.images):
            raise ValueError("image ids must be unique within an example")
        if self.kind not in ("noul", "choice", "score"):
            raise ValueError(f"unknown kind: {self.kind}")
        if not 0 <= self.label < len(self.options):
            raise ValueError(f"label {self.label} out of range for {len(self.options)} options")

    @property
    def n_options(self) -> int:
        return len(self.options)

    def all_instructions(self) -> list[str]:
        """Canonical phrasing first, then any alternatives."""
        return [self.instructions, *self.instruction_variants]

    def to_question(self, instructions: str | None = None) -> Question:
        """Rebuild the API-level question, so training and serving share one renderer.

        `instructions` overrides the canonical phrasing, which is how the collator
        feeds a sampled variant through without mutating the stored example.
        """
        instr = self.instructions if instructions is None else instructions
        crit = {name: desc for name, desc in self.options}
        if self.kind == "noul":
            return NoulQuestion(instructions=instr, criteria=crit)
        if self.kind == "choice":
            return ChoiceQuestion(instructions=instr, criteria=crit)
        return ScoreQuestion(instructions=instr, criteria=[desc for _, desc in self.options])

    def to_json(self) -> str:
        data = asdict(self)
        if not self.images:
            data.pop("images")
        if self.document_id is None:
            data.pop("document_id")
        return json.dumps(data, ensure_ascii=False)

    def to_images(self) -> list[ImageInput]:
        return [image.to_input() for image in self.images]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(
            kind=d["kind"],
            state=d["state"],
            instructions=d["instructions"],
            options=[list(o) for o in d["options"]],
            label=int(d["label"]),
            task=d.get("task", "unknown"),
            weight=float(d.get("weight", 1.0)),
            instruction_variants=list(d.get("instruction_variants") or []),
            images=[ImageAsset(**image) for image in d.get("images", [])],
            document_id=d.get("document_id"),
        )


def _open(path: str, mode: str) -> IO[str]:
    """Transparent gzip so large corpora do not need a separate decompress step."""
    if path.endswith(".gz"):
        return cast(IO[str], gzip.open(path, mode + "t", encoding="utf-8"))
    return open(path, mode, encoding="utf-8")


def write_jsonl(path: str, examples: Iterable[Example]) -> int:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    n = 0
    with _open(path, "w") as fh:
        for ex in examples:
            fh.write(ex.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[Example]:
    with _open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                example = Example.from_dict(json.loads(line))
                for image in example.images:
                    image.resolve(Path(path).resolve().parent)
                yield example


def load_examples(paths: list[str]) -> list[Example]:
    out: list[Example] = []
    for p in paths:
        out.extend(read_jsonl(p))
    return out


def split_examples(
    examples: list[Example],
    *,
    val_fraction: float = 0.05,
    seed: int = 0,
    group_by_task: bool = True,
) -> tuple[list[Example], list[Example]]:
    """Split train/val, stratified by task by default.

    Stratifying matters here: the corpus mixes tasks with wildly different sizes,
    and an unstratified 5% slice can miss a small task entirely, which then shows
    up as a misleading per-task eval hole.
    """
    if any(ex.images or ex.document_id for ex in examples):
        groups = grouped_examples(examples)
        random.Random(seed).shuffle(groups)
        cut = max(1, int(len(groups) * (1 - val_fraction))) if len(groups) > 1 else 1
        return (
            [ex for group in groups[:cut] for ex in group],
            [ex for group in groups[cut:] for ex in group],
        )
    rng = random.Random(seed)
    if not group_by_task:
        shuffled = list(examples)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * (1 - val_fraction))
        return shuffled[:cut], shuffled[cut:]

    by_task: dict[str, list[Example]] = {}
    for ex in examples:
        by_task.setdefault(ex.task, []).append(ex)

    train: list[Example] = []
    val: list[Example] = []
    for task in sorted(by_task):
        items = by_task[task]
        rng.shuffle(items)
        cut = max(1, int(len(items) * (1 - val_fraction))) if len(items) > 1 else 1
        train.extend(items[:cut])
        val.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def held_out_task_split(
    examples: list[Example], held_out: list[str]
) -> tuple[list[Example], list[Example]]:
    """Split by task name, for measuring the thing that actually matters.

    In-distribution accuracy on tasks the model trained on says little about whether
    the generic heads learned to bind to prompt-described options. Holding out whole
    tasks -- unseen label sets, unseen domains -- is the honest test of that.
    """
    hold = set(held_out)
    seen = [e for e in examples if e.task not in hold]
    unseen = [e for e in examples if e.task in hold]
    return seen, unseen


def group_keys(ex: Example) -> set[str]:
    if ex.images and (ex.document_id is None or not ex.document_id.strip()):
        raise ValueError("visual rows require document_id for grouped splits")
    keys = {f"document:{ex.document_id}"} if ex.document_id else set()
    for image in ex.images:
        keys.add(f"image-sha256:{image.sha256}")
        if image.source_id:
            keys.add(f"source:{image.source_id}")
    return keys


def grouped_examples(examples: list[Example]) -> list[list[Example]]:
    """Connected components: linked crops, documents and source variants stay together."""
    parents = list(range(len(examples)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    owners: dict[str, int] = {}
    for index, ex in enumerate(examples):
        for key in group_keys(ex):
            if key in owners:
                parents[find(index)] = find(owners[key])
            else:
                owners[key] = index
    groups: dict[int, list[Example]] = {}
    for index, ex in enumerate(examples):
        groups.setdefault(find(index), []).append(ex)
    return list(groups.values())


def assert_disjoint(**splits: list[Example]) -> None:
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        keys = {key for row in rows for key in group_keys(row)}
        for key in sorted(keys):
            if key in seen:
                raise ValueError(
                    f"document/source/content overlap between {seen[key]} and {name}: {key}"
                )
            seen[key] = name
