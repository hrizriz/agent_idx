"""Sync IDX daily_stock_summary parquet files from a public Google Drive folder.

Resume-safe: skips files already on disk. Uses gdown with backoff.

Default folder: idx_parquets (Ringkasan Saham daily panels with foreign flow).

Usage:
    py -3 scripts/sync_drive_idx_panel.py
    py -3 scripts/sync_drive_idx_panel.py --limit 50
    py -3 scripts/sync_drive_idx_panel.py --publish   # copy into data/parquet/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_FOLDER_ID = "1pCg-n2-DLugPNxu3FrVhIAQUTKVOYbdI"
DEFAULT_OUT = ROOT / "data" / "parquet" / "drive_idx"
PARQUET_DIR = ROOT / "data" / "parquet"


def _flush(msg: str) -> None:
    print(msg, flush=True)


def list_folder(folder_id: str, out_dir: Path) -> list[dict]:
    import gdown

    _flush(f"Listing Drive folder {folder_id} ...")
    items = gdown.download_folder(
        id=folder_id,
        output=str(out_dir),
        quiet=False,
        use_cookies=True,
        skip_download=True,
        resume=True,
    )
    files: list[dict] = []
    for it in items or []:
        fid = getattr(it, "id", None)
        path = getattr(it, "path", None) or getattr(it, "local_path", None)
        name = Path(path).name if path else getattr(it, "name", None)
        if fid and name and str(name).endswith(".parquet"):
            files.append({"id": str(fid), "name": Path(str(name)).name})
    manifest = out_dir / "_manifest.json"
    manifest.write_text(json.dumps(files, indent=2), encoding="utf-8")
    _flush(f"Listed {len(files)} parquet files -> {manifest}")
    return files


def download_one(file_id: str, dest: Path) -> Path:
    """Download one Drive file via uc?export=download&confirm=t (small parquet-friendly)."""
    import requests

    url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t"
    with requests.get(url, timeout=120, allow_redirects=True, stream=True) as resp:
        resp.raise_for_status()
        raw = resp.content
    if len(raw) < 1000:
        raise RuntimeError(f"too small ({len(raw)} bytes)")
    head = raw[:200].lstrip().lower()
    if head.startswith(b"<!doctype") or head.startswith(b"<html"):
        raise RuntimeError("got HTML instead of parquet (rate-limit/permission)")
    dest.write_bytes(raw)
    return dest


def download_missing(
    files: list[dict],
    out_dir: Path,
    *,
    limit: int = 0,
    sleep_sec: float = 1.5,
) -> tuple[int, int]:
    have = {p.name for p in out_dir.glob("*.parquet")}
    todo = [f for f in files if f["name"] not in have]
    # Prefer recent names first (YYYY-MM-DD and YYYYMMDD)
    todo.sort(key=lambda f: f["name"], reverse=True)
    if limit > 0:
        todo = todo[:limit]
    _flush(f"Download queue: {len(todo)} (already have {len(have)})")

    ok_n = fail_n = 0
    for idx, f in enumerate(todo, 1):
        dest = out_dir / f["name"]
        success = False
        for attempt in range(1, 8):
            try:
                download_one(f["id"], dest)
                if dest.is_file() and dest.stat().st_size > 1000:
                    ok_n += 1
                    success = True
                    break
                raise RuntimeError(
                    f"bad size {dest.stat().st_size if dest.exists() else 0}"
                )
            except Exception as exc:  # noqa: BLE001
                wait = min(120, 4 * (2 ** (attempt - 1)))
                _flush(
                    f"[{idx}/{len(todo)}] retry {attempt}/7 {f['name']}: "
                    f"{type(exc).__name__}: {str(exc)[:140]} sleep={wait}s"
                )
                time.sleep(wait)
                if dest.exists() and dest.stat().st_size < 1000:
                    dest.unlink(missing_ok=True)
        if not success:
            fail_n += 1
            _flush(f"FAIL {f['name']}")
        elif idx % 10 == 0 or idx == len(todo):
            _flush(f"progress {idx}/{len(todo)} ok={ok_n} fail={fail_n}")
        time.sleep(max(0.0, sleep_sec))
    return ok_n, fail_n


def normalize_panel_filename(name: str) -> str | None:
    """daily_stock_summary_2026-07-31.parquet -> daily_stock_summary_20260731.parquet."""
    import re

    m = re.match(
        r"daily_stock_summary_(\d{4})-?(\d{2})-?(\d{2})\.parquet$",
        name,
        re.I,
    )
    if not m:
        return None
    return f"daily_stock_summary_{m.group(1)}{m.group(2)}{m.group(3)}.parquet"


def publish_to_parquet_dir(
    src_dir: Path, dest_dir: Path = PARQUET_DIR
) -> tuple[int, int]:
    """Copy Drive panels into data/parquet with normalized YYYYMMDD names."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for src in sorted(src_dir.glob("daily_stock_summary_*.parquet")):
        norm = normalize_panel_filename(src.name)
        if not norm:
            skipped += 1
            continue
        dest = dest_dir / norm
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        dest.write_bytes(src.read_bytes())
        copied += 1
    return copied, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder-id", default=DEFAULT_FOLDER_ID)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=1.5)
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only refresh _manifest.json",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Copy downloaded files into data/parquet/ (normalized names)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download; use existing drive_idx + optional --publish",
    )
    parser.add_argument(
        "--refresh-manifest",
        action="store_true",
        help="Re-list Drive folder even if _manifest.json exists",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "_manifest.json"
    if manifest_path.is_file() and not args.refresh_manifest:
        files = json.loads(manifest_path.read_text(encoding="utf-8"))
        _flush(f"Loaded manifest n={len(files)}")
        if not files:
            files = list_folder(args.folder_id, out_dir)
    else:
        files = list_folder(args.folder_id, out_dir)

    if args.list_only:
        return 0

    if not args.skip_download:
        ok_n, fail_n = download_missing(
            files, out_dir, limit=args.limit, sleep_sec=args.sleep
        )
        _flush(
            f"Download done ok={ok_n} fail={fail_n} "
            f"on_disk={len(list(out_dir.glob('*.parquet')))}"
        )
        if fail_n and ok_n == 0:
            return 1

    if args.publish:
        copied, skipped = publish_to_parquet_dir(out_dir)
        _flush(f"Published to {PARQUET_DIR}: copied={copied} skipped={skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
