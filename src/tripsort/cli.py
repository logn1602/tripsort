"""Command-line interface: a thin Typer layer over the tripsort API.

Exit codes: 0 success, 1 user error, 2 partial success (some files unreadable, listed in
the summary and in manifest.json).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    MofNCompleteColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.progress import Progress as RichProgressBar
from typer.core import TyperGroup

from tripsort import __version__
from tripsort.stats import count  # standard library only, so --version stays fast

if TYPE_CHECKING:
    from tripsort.api import AnalyzeResult, Project
    from tripsort.export.writer import ExportResult
    from tripsort.faces.people import RefsResult
    from tripsort.ingest.pipeline import IngestResult

EXIT_OK = 0
EXIT_USER_ERROR = 1
EXIT_PARTIAL = 2


@contextmanager
def _usage_errors_exit_1() -> Iterator[None]:
    """Typer exits with 2 on usage errors; tripsort keeps 2 for partial success."""
    try:
        yield
    except Exception as exc:
        if getattr(exc, "exit_code", None) == EXIT_PARTIAL and not isinstance(exc, typer.Exit):
            exc.exit_code = EXIT_USER_ERROR  # type: ignore[attr-defined]
        raise


class _Group(TyperGroup):
    def make_context(self, *args: Any, **kwargs: Any) -> Any:
        with _usage_errors_exit_1():
            return super().make_context(*args, **kwargs)

    def invoke(self, ctx: Any) -> Any:
        with _usage_errors_exit_1():
            return super().invoke(ctx)


app = typer.Typer(
    name="tripsort",
    cls=_Group,
    help="Turn a group trip's photo zips into deduplicated, sorted folders, fully offline.",
    no_args_is_help=True,
    add_completion=False,
)

err = Console(stderr=True, highlight=False, soft_wrap=True)


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


ZipsArg = Annotated[
    list[Path],
    typer.Argument(metavar="ZIP", help="Zip files of photos and videos, usually one per person."),
]
ProjectArg = Annotated[Path, typer.Argument(metavar="PROJECT", help="The project (output) folder.")]
RefsOpt = Annotated[
    Path | None,
    typer.Option(
        help="Folder with one clear photo per person (NAME.jpg or NAME/*.jpg), used to name people."
    ),
]
CopyOpt = Annotated[
    bool, typer.Option("--copy", help="Make copies instead of hardlinks in the review folders.")
]
DryRunOpt = Annotated[
    bool, typer.Option("--dry-run", help="Show what the export would do; write no output files.")
]
DuplicatesOpt = Annotated[
    bool,
    typer.Option(
        "--export-duplicates",
        help="Also write duplicates/: each kept file next to the copies left out of all/.",
    ),
]
BenchmarkOpt = Annotated[
    bool, typer.Option("--benchmark", help="Print time, files per second, and peak memory.")
]


@app.command()
def run(
    zips: ZipsArg,
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output folder for the sorted files and the index."),
    ],
    refs: RefsOpt = None,
    copy: CopyOpt = False,
    dry_run: DryRunOpt = False,
    export_duplicates: DuplicatesOpt = False,
    benchmark: BenchmarkOpt = False,
) -> None:
    """Ingest, analyze, and export in one step."""
    _check_refs(refs)
    with _project(out) as project:
        ingested = project.ingest(zips, progress=_RichProgress())
        _show_ingest(ingested)
        if refs is not None and ingested.face_problem:
            err.print("[yellow]warning:[/] --refs is not used: the face models are not available.")
            refs = None
        analyzed = _analyze_with_refs(project, refs)
        typer.echo(analyzed.summary(project=str(out)))
        exported = project.export(
            hardlinks=not copy,
            dry_run=dry_run,
            export_duplicates=export_duplicates,
            progress=_RichProgress(),
        )
        typer.echo(exported.summary())
        if benchmark:
            _show_benchmark(ingested, analyzed, exported)
        code = _exit_code(project, ingested, exported)
    raise typer.Exit(code)


@app.command()
def ingest(
    project_dir: ProjectArg,
    zips: Annotated[
        list[Path] | None,
        typer.Argument(
            metavar="[ZIP]...",
            help="Zips to add. With none, only photos not yet scanned for faces are scanned.",
        ),
    ] = None,
    benchmark: BenchmarkOpt = False,
) -> None:
    """Add zips to a project: hash every file, and decode and scan for faces only new ones."""
    with _project(project_dir) as project:
        result = project.ingest(zips or [], progress=_RichProgress())
        _show_ingest(result)
        if benchmark:
            _show_benchmark(result, None, None)
    partial = result.problems or _models_missing(result)
    raise typer.Exit(EXIT_PARTIAL if partial else EXIT_OK)


@app.command()
def analyze(project_dir: ProjectArg, refs: RefsOpt = None, benchmark: BenchmarkOpt = False) -> None:
    """Resolve dates, find copies, and sort faces into people (reads only the index)."""
    _check_refs(refs)
    with _project(project_dir) as project:
        result = _analyze_with_refs(project, refs)
        typer.echo(result.summary(project=str(project_dir)))
        if benchmark:
            _show_benchmark(None, result, None)


@app.command()
def export(
    project_dir: ProjectArg,
    copy: CopyOpt = False,
    dry_run: DryRunOpt = False,
    export_duplicates: DuplicatesOpt = False,
    benchmark: BenchmarkOpt = False,
) -> None:
    """Write all/, the review folders, and manifest.json."""
    with _project(project_dir) as project:
        result = project.export(
            hardlinks=not copy,
            dry_run=dry_run,
            export_duplicates=export_duplicates,
            progress=_RichProgress(),
        )
        typer.echo(result.summary())
        if benchmark:
            _show_benchmark(None, None, result)
        code = _exit_code(project, None, result)
    raise typer.Exit(code)


@app.command()
def clean(
    project_dir: ProjectArg,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation.")] = False,
) -> None:
    """Delete the project's index (.tripsort/: hashes, dates, camera and GPS data, faces,
    face embeddings, and names) and the review pages (review/, which show face crops).

    The exported folders and manifest.json are kept.
    """
    from tripsort.api import clean as clean_index
    from tripsort.faces.review import REVIEW_DIR
    from tripsort.index.db import INDEX_DIR_NAME

    targets = [p for p in (project_dir / INDEX_DIR_NAME, project_dir / REVIEW_DIR) if p.exists()]
    if not targets:
        typer.echo(f"Nothing to clean: {project_dir / INDEX_DIR_NAME} does not exist.")
        return
    listed = " and ".join(str(t) for t in targets)
    if not yes:
        typer.confirm(f"Delete {listed}?", abort=True)
    with _user_errors():
        clean_index(project_dir)
    typer.echo(f"Deleted {listed}.")


people_app = typer.Typer(
    cls=_Group,
    help="List, review, rename, merge, and keep apart the people found in a project.",
    no_args_is_help=True,
)
app.add_typer(people_app, name="people")


@people_app.command("list")
def people_list(project_dir: ProjectArg) -> None:
    """Show everyone found, most photos first, then the suggested merges."""
    from tripsort.faces.suggest import suggestion_lines

    with _project(project_dir) as project:
        people = project.people.all()
        suggestions = project.people.suggestions()
    if not people:
        typer.echo("No people found yet.")
        return
    width = max(len("NAME"), *(len(p.name) for p in people))
    typer.echo(f"{'NAME':<{width}}  {'PHOTOS':>6}  {'SOLO':>5}  {'FACES':>6}  NAMED BY")
    for p in people:
        named_by = {"refs": "refs/", "user": "you"}.get(p.named_by or "", "-")
        typer.echo(f"{p.name:<{width}}  {p.photos:>6,}  {p.solo:>5,}  {p.faces:>6,}  {named_by}")
    lines = suggestion_lines(suggestions, str(project_dir))
    if lines:
        typer.echo("")
        typer.echo("\n".join(lines))


@people_app.command("review")
def people_review(
    project_dir: ProjectArg,
    per_person: Annotated[
        int, typer.Option(min=1, help="Sample faces shown for each person.")
    ] = 24,
) -> None:
    """Write review/people.html: the suggested merges with every face of the person
    suggested, and sample face crops for each person, to check before sharing."""
    with _project(project_dir) as project:
        path = project.people.review(
            per_person=per_person, progress=_RichProgress(), project=str(project_dir)
        )
    typer.echo(f"Wrote {path}")
    typer.echo("It shows faces, so keep it private; `tripsort clean` deletes it with the index.")


@people_app.command("rename")
def people_rename(
    project_dir: ProjectArg,
    person: Annotated[str, typer.Argument(help="Current name, or person_NN.")],
    new_name: Annotated[str, typer.Argument(metavar="NEW_NAME", help="The name to give.")],
) -> None:
    """Name a person (or rename them). Run `tripsort export` afterwards to move the folder."""
    with _project(project_dir) as project:
        info = project.people.rename(person, new_name)
    typer.echo(f"person_{info.number:02d} is now {info.name} ({count(info.photos, 'photo')}).")


@people_app.command("merge")
def people_merge(
    project_dir: ProjectArg,
    person: Annotated[str, typer.Argument(help="The person to merge away.")],
    into: Annotated[str, typer.Argument(help="The person who keeps every photo.")],
) -> None:
    """Say two people are one, for example one person split by sunglasses."""
    with _project(project_dir) as project:
        info = project.people.merge(person, into)
    typer.echo(f"Merged {person} into {info.name}, who now has {count(info.photos, 'photo')}.")


@people_app.command("dismiss")
def people_dismiss(
    project_dir: ProjectArg,
    person: Annotated[str, typer.Argument(help="A name, or person_NN.")],
    other: Annotated[str, typer.Argument(help="The person they are not.")],
) -> None:
    """Say two people are not one: their merge is never suggested again, and later
    clusterings keep them apart."""
    with _project(project_dir) as project:
        first, second = project.people.dismiss(person, other)
    typer.echo(
        f"Keeping {first.name} and {second.name} apart; "
        "tripsort will not suggest merging them again."
    )


models_app = typer.Typer(
    cls=_Group, help="The face models tripsort downloads on first use.", no_args_is_help=True
)
app.add_typer(models_app, name="models")


@models_app.command("list")
def models_list() -> None:
    """Show each model, its license, and whether it is downloaded."""
    from tripsort.models.download import model_dir, model_status
    from tripsort.models.registry import ALL_MODELS
    from tripsort.stats import human_bytes

    typer.echo(f"Model folder: {model_dir()}")
    for m in ALL_MODELS:
        typer.echo(
            f"  {m.name:<6} {m.version:<8} {human_bytes(m.size):>8}  {m.license:<10} "
            f"{model_status(m):<8} {m.purpose}"
        )


@models_app.command("download")
def models_download() -> None:
    """Download and verify every model now, for example before going offline."""
    from tripsort.models.download import ensure_models
    from tripsort.models.registry import ALL_MODELS

    with _user_errors():
        paths = ensure_models(ALL_MODELS, _RichProgress())
    for name, path in paths.items():
        typer.echo(f"{name}: {path}")


@contextmanager
def _user_errors() -> Iterator[None]:
    from tripsort.errors import UserError

    try:
        yield
    except UserError as exc:
        err.print(f"[red]error:[/] {exc}")
        raise typer.Exit(EXIT_USER_ERROR) from None
    except OSError as exc:
        err.print(f"[red]error:[/] {exc}")
        raise typer.Exit(EXIT_USER_ERROR) from None


@contextmanager
def _project(folder: Path) -> Iterator["Project"]:
    from tripsort.api import Project

    with _user_errors():
        project = Project.open(folder)
        try:
            yield project
        finally:
            project.close()


def _check_refs(refs: Path | None) -> None:
    if refs is not None and not refs.is_dir():
        err.print(f"[red]error:[/] {refs}: no such folder")
        raise typer.Exit(EXIT_USER_ERROR)


def _analyze_with_refs(project: "Project", refs: Path | None) -> "AnalyzeResult":
    """Analyze; with ``refs``, embed the reference photos first so they name people."""
    if refs is None:
        return project.analyze()
    with _user_errors():
        named = project.people.name_from_refs(refs)
    _show_refs(named)
    return named.analysis


def _show_refs(result: "RefsResult") -> None:
    for problem in result.problems:
        err.print(f"[yellow]refs: {problem}[/]")
    typer.echo(result.summary())


def _exit_code(
    project: "Project", ingested: "IngestResult | None", exported: "ExportResult"
) -> int:
    stats = project.stats()
    partial = (
        bool(ingested and (ingested.problems or _models_missing(ingested)))
        or bool(exported.problems)
        or stats.unreadable > 0
        or stats.read_failures > 0
    )
    return EXIT_PARTIAL if partial else EXIT_OK


def _models_missing(result: "IngestResult") -> bool:
    """Some photos could not be scanned for faces or embedded with CLIP."""
    return bool(
        result.face_problem or result.faces_pending or result.clip_problem or result.clip_pending
    )


def _show_ingest(result: "IngestResult") -> None:
    typer.echo(result.summary())
    for problem in result.problems:
        err.print(f"[yellow]{problem.source}: {problem.member}[/]: {problem.message}")


def _show_benchmark(
    ingested: "IngestResult | None",
    analyzed: "AnalyzeResult | None",
    exported: "ExportResult | None",
) -> None:
    from tripsort.stats import human_bytes

    lines = ["Benchmark:"]
    peaks: list[int] = []
    rows: list[tuple[str, int, int, float, int | None]] = []
    if ingested is not None:
        i = ingested
        rows.append(("ingest", i.files_read, i.bytes_read, i.seconds, i.peak_rss_bytes))
    if analyzed is not None:
        a = analyzed
        rows.append(("analyze", a.stats.unique, 0, a.seconds, a.peak_rss_bytes))
    if exported is not None:
        e = exported
        rows.append(("export", e.written, e.bytes_written, e.seconds, e.peak_rss_bytes))
    for stage, items, size, seconds, peak in rows:
        rate = f"{items / seconds:,.1f} files/s" if seconds > 0 else "-"
        speed = f", {human_bytes(int(size / seconds))}/s" if size and seconds > 0 else ""
        lines.append(f"  {stage:<8} {items:>7,} files  {seconds:8.2f} s  {rate}{speed}")
        if peak is not None:
            peaks.append(peak)
    if peaks:
        lines.append(f"  peak memory: {human_bytes(max(peaks))}")
    typer.echo("\n".join(lines))


class _RichProgress:
    """Draws the API's progress hooks as a Rich bar with an ETA (on terminals only)."""

    def __init__(self) -> None:
        self._bar: RichProgressBar | None = None
        self._task: Any = None

    def start(self, stage: str, total: int, unit: str) -> None:
        if not err.is_terminal or total <= 0:
            return
        columns: list[Any] = [TextColumn("{task.description:<7}"), BarColumn()]
        if unit == "bytes":
            columns += [DownloadColumn(), TransferSpeedColumn()]
        else:
            columns += [MofNCompleteColumn()]
        columns.append(TimeRemainingColumn())
        self._bar = RichProgressBar(*columns, console=err, transient=True)
        self._bar.start()
        self._task = self._bar.add_task(stage, total=total)

    def advance(self, amount: int) -> None:
        if self._bar is not None:
            self._bar.advance(self._task, amount)

    def finish(self) -> None:
        if self._bar is not None:
            self._bar.stop()
            self._bar = None
