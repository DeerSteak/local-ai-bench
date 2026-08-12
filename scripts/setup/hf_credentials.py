"""Hugging Face credential loading and optional persistence for setup."""

import os
from pathlib import Path

from scripts.setup.setup_console import CYAN, RESET, YELLOW, confirm, link, ok, warn
from scripts.setup.setup_selection import save_hf_token


class HfTokenProvider:
    def __init__(self, root: Path, gated_models_selected: bool):
        self.root = root
        self.gated_models_selected = gated_models_selected
        self._loaded = False
        self._token = ""

    def set(self, token: str | None) -> None:
        self._loaded = True
        self._token = (token or "").strip()

    def load_existing(self) -> str:
        token = os.environ.get("HF_TOKEN", "").strip()
        if token:
            return token
        path = self.root / "hf.txt"
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def load(self) -> str:
        if self._loaded:
            return self._token
        token = os.environ.get("HF_TOKEN", "").strip()
        if token:
            ok("HuggingFace token loaded from HF_TOKEN env var")
            self.set(token)
            return token
        token = self.load_existing()
        if token:
            ok("HuggingFace token loaded from hf.txt")
            self.set(token)
            return token
        self._print_guidance()
        skip = "skip gated models" if self.gated_models_selected else "skip and download without one"
        try:
            token = input(
                f"  {CYAN}Paste your HuggingFace token and press Enter{RESET}\n"
                f"  (or press Enter to {skip}): ",
            ).strip()
        except EOFError:
            token = ""
        if token and confirm("Save token to hf.txt for future runs?", default=True):
            try:
                save_hf_token(self.root / "hf.txt", token)
                ok("Token saved to hf.txt")
            except OSError as exc:
                warn(f"Could not save hf.txt: {exc}")
        self.set(token)
        return token

    def save(self, token: str) -> None:
        save_hf_token(self.root / "hf.txt", token)

    def _print_guidance(self) -> None:
        print()
        print(f"  {YELLOW}Models are downloaded from HuggingFace; an account is optional.{RESET}")
        print(f"  1. Create an account at {link('https://huggingface.co')}")
        step = 2
        if self.gated_models_selected:
            print("  2. Accept the selected gated-model licenses on Hugging Face")
            step = 3
        print(f"  {step}. Generate a token at {link('https://huggingface.co/settings/tokens')}")
        print()
