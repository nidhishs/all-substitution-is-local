"""Experiments CLI."""

from __future__ import annotations

import click

from .experiment_1.runner import main as experiment_1
from .experiment_3.runner import main as experiment_3


@click.group()
def main() -> None:
    """Experiments."""


main.add_command(experiment_1, name="experiment-1")
main.add_command(experiment_3, name="experiment-3")


if __name__ == "__main__":
    main()
