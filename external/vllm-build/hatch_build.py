"""Apply the vllm patches before hatchling assembles the wheel."""

import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class VllmPatchHook(BuildHookInterface):
    PLUGIN_NAME = "vllm-patch"

    def initialize(self, version: str, build_data: dict) -> None:  # noqa: ARG002
        root = Path(self.root)
        submodule = (root / ".." / "vllm").resolve()
        patches_dir = root / "patches"

        if not (submodule / ".git").exists() and not (submodule / ".git").is_file():
            raise RuntimeError(
                f"vllm submodule not initialized at {submodule}. Run `git submodule update --init external/vllm` first."
            )

        for patch in sorted(patches_dir.glob("*.patch")):
            check = subprocess.run(
                ["git", "-C", str(submodule), "apply", "--check", "--reverse", str(patch)],
                capture_output=True,
            )
            if check.returncode == 0:
                continue
            subprocess.run(
                ["git", "-C", str(submodule), "apply", str(patch)],
                check=True,
            )
