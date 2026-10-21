"""vla_compat.py — the vendored OpenVLA prompt builder (Phase 1).

Pinned verbatim to the openvla commit (see the module header); these tests
pin the exact strings the engine's training construction depends on — the
byte-for-byte match with the official PurePromptBuilder output is what keeps
our training inputs aligned with the pretrained model's expectations.
"""

from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(REPO))

from images import vla_compat  # noqa: E402


def test_vendored_source_header_pins_the_commit():
    src = pathlib.Path(vla_compat.__file__).read_text()
    assert "c8f03f48af692657d3060c19588038c7220e9af9" in src
    assert "base_prompter.py" in src  # the vendored file's provenance


def test_single_turn_prompt_exact_string():
    pb = vla_compat.PurePromptBuilder("openvla")
    pb.add_turn("human", "What action should the robot take to stack the blocks?")
    pb.add_turn("gpt", "ACTIONTOKENS")
    # wrap_human ends with a trailing space after "Out:"; wrap_gpt appends eos
    assert pb.get_prompt() == \
        "In: What action should the robot take to stack the blocks?\nOut: " \
        "ACTIONTOKENS</s>"


def test_message_strip_and_image_token_removal():
    pb = vla_compat.PurePromptBuilder("openvla")
    pb.add_turn("human", "  push the red cup <image>  ")
    pb.add_turn("gpt", "X")
    assert pb.get_prompt() == "In: push the red cup\nOut: X</s>"


def test_empty_gpt_turn_becomes_space():
    # wrap_gpt: an empty gpt message becomes " " (a single space) + eos
    pb = vla_compat.PurePromptBuilder("openvla")
    pb.add_turn("human", "do it")
    pb.add_turn("gpt", "")
    assert pb.get_prompt() == "In: do it\nOut:  </s>"


def test_multi_turn_accumulation():
    pb = vla_compat.PurePromptBuilder("openvla")
    pb.add_turn("human", "first")
    pb.add_turn("gpt", "A")
    pb.add_turn("human", "second")
    pb.add_turn("gpt", "B")
    assert pb.get_prompt() == "In: first\nOut: A</s>In: second\nOut: B</s>"
