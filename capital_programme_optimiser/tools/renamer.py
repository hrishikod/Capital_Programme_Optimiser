"""Scenario cache renaming utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from capital_programme_optimiser.config import load_settings


def prefix_cache_files(prefix: str,
                       folder: Path | None = None,
                       dry_run: bool = False) -> dict[str, int]:
    settings = load_settings()
    folder = folder or settings.cache_dir()
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    renamed = 0
    skipped_prefixed = 0
    skipped_exists = 0

    for path in sorted(folder.glob('*.pkl')):
        if path.name.startswith(prefix):
            skipped_prefixed += 1
            continue
        target = path.with_name(prefix + path.name)
        if target.exists():
            skipped_exists += 1
            continue
        if not dry_run:
            path.rename(target)
        renamed += 1

    return {
        'renamed': renamed,
        'already_prefixed': skipped_prefixed,
        'target_exists': skipped_exists,
    }


__all__ = ['prefix_cache_files']
