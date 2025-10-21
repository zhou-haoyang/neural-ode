"""Lightning CLI entry point for neural-ode repository.

Usage examples:
  python -m tools.cli fit --config config.yaml
  lightning run --config config.yaml

This module exposes a LightningCLI that constructs the `LatentODELightning` module and
allows passing a DataModule (optional) or raw Trainer configuration via CLI.
"""

from __future__ import annotations

import argparse
import os

import lightning as L
from lightning.pytorch.cli import LightningCLI

from ..models import LatentODELightning


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: argparse.ArgumentParser) -> None:
        # allow user to pass a path to checkpoint to resume
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)


def main():
    # Minimal CLI: constructs LatentODELightning by default. Users can provide a DataModule
    # or Trainer config via CLI/--config.
    cli = CLI(LatentODELightning, trainer_class=L.Trainer)


if __name__ == "__main__":
    main()
