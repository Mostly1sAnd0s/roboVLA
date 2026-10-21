"""vla_compat.py — the vendored OpenVLA prompt builder (design §6, Phase 1).

Vendored verbatim from the PINNED openvla commit
c8f03f48af692657d3060c19588038c7220e9af9 (verified 2026-10-21, host65 probe):

    prismatic/models/backbones/llm/prompting/base_prompter.py  (PurePromptBuilder)

Why vendored: the runner image does NOT ``pip install`` the openvla package.
``import prismatic`` pulls ``draccus`` (prismatic.conf), and importing
anything under ``prismatic.vla`` pulls the RLDS datasets — ``tensorflow`` +
``tensorflow_datasets`` — via ``prismatic.vla.materialize``. The engine needs
only the prompt construction (the action-token math is pure in engine.py and
mirrors the official ``ActionTokenizer`` exactly — cross-checked by the
Phase 1 probe on 200 random values). The crosscheck job brings the official
stack's dependencies at node join (Phase 1, §12).

Re-pinning the openvla commit = re-vendor from the new SHA + re-run the
probe (the probe file-loads the real modules, so any drift is caught).
"""

from abc import ABC, abstractmethod
from typing import Optional


class PromptBuilder(ABC):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        self.model_family = model_family

        # Only some models define a system prompt => let subclasses handle this logic!
        self.system_prompt = system_prompt

    @abstractmethod
    def add_turn(self, role: str, message: str) -> str: ...

    @abstractmethod
    def get_potential_prompt(self, user_msg: str) -> None: ...

    @abstractmethod
    def get_prompt(self) -> str: ...


class PurePromptBuilder(PromptBuilder):
    def __init__(self, model_family: str, system_prompt: Optional[str] = None) -> None:
        super().__init__(model_family, system_prompt)

        # TODO (siddk) =>> Can't always assume LlamaTokenizer --> FIX ME!
        self.bos, self.eos = "<s>", "</s>"

        # Get role-specific "wrap" functions
        self.wrap_human = lambda msg: f"In: {msg}\nOut: "
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.eos}"

        # === `self.prompt` gets built up over multiple turns ===
        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        # Update Prompt
        self.prompt += wrapped_message

        # Bump Turn Counter
        self.turn_count += 1

        # Return "wrapped_message" (effective string added to context)
        return wrapped_message

    def get_potential_prompt(self, message: str) -> None:
        # Assumes that it's always the user's (human) turn!
        prompt_copy = str(self.prompt)

        human_message = self.wrap_human(message)
        prompt_copy += human_message

        return prompt_copy.removeprefix(self.bos).rstrip()

    def get_prompt(self) -> str:
        # Remove prefix <bos> (if exists) because it gets auto-inserted by tokenizer!
        return self.prompt.removeprefix(self.bos).rstrip()
