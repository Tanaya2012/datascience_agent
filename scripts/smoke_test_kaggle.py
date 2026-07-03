"""
Live smoke test for the Kaggle library tools (D16). Tool-level (no LLM/quota) — it
verifies real auth + search + download against Kaggle.

Needs credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).

Run from /Users/tushar/interests:
  conda run -n dsagent python -m datascience_agent.scripts.smoke_test_kaggle
"""

from __future__ import annotations

import asyncio
from pathlib import Path


async def main() -> int:
    from datascience_agent.tools.kaggle_tool import (
        download_kaggle,
        kaggle_credentials_available,
        search_kaggle,
    )

    if not kaggle_credentials_available():
        print("SKIP: no Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).")
        return 0

    print("→ search_kaggle('titanic', source='dataset')")
    s = await search_kaggle("titanic", source="dataset", limit=3)
    if not s["success"]:
        print("SEARCH FAILED:", s["error_message"])
        return 1
    for h in s["hits"]:
        print(f"   {h['ref']:35} {h.get('title')}")

    ref = "heptapod/titanic"  # small (~11 KB) public dataset
    print(f"\n→ download_kaggle('{ref}', source='dataset')")
    d = await download_kaggle(ref, source="dataset")
    if not d["success"]:
        print("DOWNLOAD FAILED:", d["error_message"])
        return 1
    print("   dir:  ", d["download_dir"])
    print("   files:", [Path(f).name for f in d["files"]])

    ok = bool(d["files"]) and Path(d["download_dir"]).is_dir()
    print(f"\nKAGGLE SMOKE {'PASSED' if ok else 'INCOMPLETE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
