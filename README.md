# tripsort

Offline Python library + CLI that merges trip photo zips, removes duplicates, and sorts photos by person, shot type, and category.

> **Status: pre-alpha.** 0.0.1 is a placeholder release: the package installs and
> `tripsort --version` works, but nothing else does yet. The first useful pre-release
> will be 0.1.0a1 (ingest, index, and dedupe).

## What it will do

After a group trip, everyone has a zip of photos from their own phone. tripsort takes all of them and:

- merges them into one pool and removes exact duplicates and recompressed copies (such as WhatsApp copies), keeping the best original;
- groups faces by person across every zip, so each friend gets a folder with every photo they appear in;
- splits shots into solo, group, and selfie, ignoring strangers in the background;
- sorts photos into categories such as scenery, animals, food, and documents, defined as editable text prompts;
- culls bursts down to the best shot and flags blurry, dark, or eyes-closed photos without deleting anything;
- exports plain folders plus a `manifest.json`, and never modifies the input zips.

Everything runs locally on an ordinary laptop CPU: no account, no cloud uploads, no telemetry.

## Planned usage

Not implemented yet; this is the target interface:

```text
uvx tripsort run mine.zip sam.zip alex.zip -o sorted/ --refs refs/

sorted/
├── all/            every kept file, once
├── people/         one folder per person
├── shots/          solo/, group/, selfie/
├── categories/     scenery/, animals/, food/, documents/
├── rejects/        blurry, dark, eyes closed: review before deleting
└── manifest.json
```

```python
from tripsort import Project

proj = Project.open("sorted/")
proj.ingest(["mine.zip", "sam.zip", "alex.zip"])
proj.analyze()
result = proj.export()
```

## Install

Requires Python 3.11 or newer.

```console
uvx tripsort --version
```

or `pip install tripsort`.

## Development

The project uses [uv](https://docs.astral.sh/uv/) for everything. The design and build plan are in
[docs/SPEC.md](docs/SPEC.md), and settled decisions are logged in [docs/decisions.md](docs/decisions.md).

```console
uv sync                              # create .venv with the dev tools
uv run pytest --cov                  # tests with coverage
uv run ruff check .                  # lint
uv run ruff format --check .         # formatting
uv run mypy                          # strict type check of src/
uv run pre-commit install            # optional: run the checks on every commit
```

CI runs the tests, linters, and type checks on Windows, macOS, and Linux with Python 3.11 to 3.13.

### Releasing

1. Set the new version in `pyproject.toml` (`uv version <new-version>`), merge to `main`.
2. Tag the merge commit on `main` with `v<version>` and push the tag.
3. `.github/workflows/release.yml` builds the package, publishes it to TestPyPI, installs it from
   there on every OS (only tripsort from TestPyPI, its dependencies from PyPI), then publishes to
   PyPI. Both uploads use Trusted Publishing, so the repo stores no tokens.
4. To repeat the TestPyPI install check locally: `uv run tools/verify_testpypi.py`.

## License

MIT. See [LICENSE](LICENSE).
