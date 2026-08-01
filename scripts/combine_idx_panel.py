"""Publish Drive IDX panels + fill missing days from tvkit chart 1D.

Usage:
    py -3 scripts/combine_idx_panel.py
    py -3 scripts/combine_idx_panel.py --no-publish-drive
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent_idx.panel_sync import combine_panels  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-publish-drive",
        action="store_true",
        help="Only rebuild tvkit gap (assume Drive already published)",
    )
    args = parser.parse_args()
    print(combine_panels(publish_drive=not args.no_publish_drive))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
