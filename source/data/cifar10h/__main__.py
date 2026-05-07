"""CIFAR-10H pipeline CLI: infer / prepare."""

from __future__ import annotations

import click

from .inference import main as infer
from .prepare import main as prepare


@click.group()
def main() -> None:
    """CIFAR-10H data pipeline."""


main.add_command(infer, name="infer")
main.add_command(prepare, name="prepare")


if __name__ == "__main__":
    main()
