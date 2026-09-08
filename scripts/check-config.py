#!/usr/bin/env python3
"""Validate the release pins and JSON embedded in the default Compose profile."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def settings(name):
    return dict(
        line.split("=", 1)
        for line in (ROOT / name).read_text().splitlines()
        if line and not line.startswith("#")
    )


def main():
    release = settings("release.env")
    env = settings(".env.example")
    assert env["IMAGE"] == release["DEFAULT_IMAGE"], "build/start image defaults differ"
    assert re.fullmatch(r"[0-9a-f]{40}", release["VLLM_COMMIT"])
    for pin in ("MODEL_REVISION", "DRAFT_REVISION"):
        assert re.fullmatch(r"[0-9a-f]{40}", env[pin]), f"unresolved model pin: {pin}"
    series = (ROOT / "patches/series").read_text().splitlines()
    assert len(series) == len(set(series)) and series, "duplicate or empty patch series"
    tracked = {
        line.split(maxsplit=1)[1]
        for line in (ROOT / "SHA256SUMS").read_text().splitlines()
    }
    for patch in series:
        assert f"patches/{patch}" in tracked, f"patch missing from checksums: {patch}"
        assert (ROOT / "patches" / patch).is_file()
    compose = (ROOT / "compose.yaml").read_text()

    def expand(match):
        key, _, default = match.group(1).partition(":-")
        return env.get(key) or default

    configs = [
        json.loads(re.sub(r"\$\{([^{}]+)\}", expand, value))
        for value in re.findall(r"^      - '(\{.*\})'$", compose, re.MULTILINE)
    ]
    spec = next(c for c in configs if "method" in c)
    assert spec == {
        "method": "dflash",
        "model": "/models/draft",
        "num_speculative_tokens": 7,
        "kv_cache_dtype": "nvfp4",
        "enforce_eager": True,
    }
    assert {"backend": "FLASHINFER"} in configs
    assert 'VLLM_FLASHINFER_XQA_USE_ISOLATED_STREAM: "1"' in compose
    for removed in (
        "DYNAMIC_SCHEDULE",
        "VLLM_DFLASH_FORCE_EAGER",
        "VLLM_XQA_DEDICATED_STREAM",
    ):
        assert removed not in compose and removed not in env
    print("release configuration: PASS")


if __name__ == "__main__":
    main()
