"""Data pipeline CLI."""

from __future__ import annotations

import click

from paths import DATA_PREPARED
from utils import validate_pair_array


@click.group()
def main() -> None:
    """Data pipeline."""


@main.command()
def validate() -> None:
    """Validate all pair artefacts under prepared/."""
    npz_files = sorted(DATA_PREPARED.glob("**/pairs/*.npz"))
    errors: list[str] = []
    for npz in npz_files:
        try:
            validate_pair_array(npz, npz.with_suffix(".json"))
        except Exception as exc:
            errors.append(f"{npz.relative_to(DATA_PREPARED)}: {exc}")
    for e in errors:
        click.echo(f"[FAIL] {e}")
    click.echo(f"[OK] {len(npz_files) - len(errors)}/{len(npz_files)} validated.")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
