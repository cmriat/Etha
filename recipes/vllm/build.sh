#!/usr/bin/env bash
set -euo pipefail

# Native extensions (.so) ship in the separate `vllm-ext` conda package, so
# vllm itself is built as a pure-Python wheel. Replace upstream's
# setuptools/CMake-driven pyproject.toml with a minimal hatchling config.
cat > pyproject.toml <<'EOF'
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "vllm"
version = "0.20.2"

[project.scripts]
vllm = "vllm.entrypoints.cli.main:main"

[project.entry-points."vllm.general_plugins"]
lora_filesystem_resolver = "vllm.plugins.lora_resolvers.filesystem_resolver:register_filesystem_resolver"
lora_hf_hub_resolver = "vllm.plugins.lora_resolvers.hf_hub_resolver:register_hf_hub_resolver"

[tool.hatch.build.targets.wheel]
packages = ["vllm"]
exclude = ["**/*.so"]
EOF

$PYTHON -m pip install . --no-build-isolation --no-deps -vv

# Conda activate hook for vllm runtime env vars. No deactivate hook —
# pixi workflows don't reliably round-trip through deactivate, and
# unwinding PATH-like vars cleanly isn't worth the machinery.
mkdir -p "$PREFIX/etc/conda/activate.d"
cp "${RECIPE_DIR}/scripts/activate.sh" "$PREFIX/etc/conda/activate.d/vllm.sh"
