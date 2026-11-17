"""Command-line interface: a thin Typer layer over the tripsort API."""

from pathlib import Path
from typing import Annotated

import typer

from tripsort import __version__

app = typer.Typer(
    name="tripsort",
    help="Turn a group trip's photo zips into deduplicated, sorted folders, fully offline.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"tripsort {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Turn a group trip's photo zips into deduplicated, sorted folders, fully offline."""


@app.command()
def run(
    zips: Annotated[
        list[Path],
        typer.Argument(
            metavar="ZIP", help="Zip files of photos and videos, usually one per person."
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output folder for the sorted files and the index."),
    ],
    refs: Annotated[
        Path | None,
        typer.Option(help="Folder with one clear photo per person, used to name people."),
    ] = None,
) -> None:
    """Ingest, analyze, and export in one step."""
    typer.echo("tripsort run is not implemented yet; it arrives in a later release.", err=True)
    raise typer.Exit(code=1)
