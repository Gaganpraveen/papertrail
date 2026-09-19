# Reproducing the tested environment

`requirements-tested.txt` records the installed runtime and development dependency versions from **CPython 3.12.14, macOS 26.6.2, arm64**, captured on 2026-09-19. Each pin has markers limiting it to Python 3.12 on macOS arm64. It is an environment snapshot, not a universal or hash-verified lock. It excludes the editable project, pip, setuptools, and wheel, and contains no local paths or platform-specific wheel URLs.

On a matching machine, create a Python 3.12 environment and constrain installation to the snapshot:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -c requirements-tested.txt -e '.[dev]'
.venv/bin/python -m pip check
.venv/bin/pytest
```

For Python 3.11, Python 3.13, or another OS/architecture, install normally with `python -m pip install -e '.[dev]'`. The snapshot's markers intentionally do not constrain those environments. The project's declared Python range remains `>=3.11,<3.14`; those other environments were not exercised by this snapshot. Installed-package metadata accepts Python 3.11 except for NumPy 2.5.3, which requires Python >=3.12. Normal dependency resolution can select a compatible NumPy release for Python 3.11; this metadata check does not replace a clean Python 3.11 installation and test run. Platform-dependent native wheels also need to be available for the chosen Python and OS.

Build isolation may select a different setuptools release. The build minimum is 77 because that release introduced support for the SPDX string form used by `license = "MIT"`. See the [setuptools license migration documentation](https://setuptools.pypa.io/en/stable/userguide/license_migration.html).

## Models and source artifacts

Dependency versions do not pin model weights. Ollama model tags may resolve to different digests over time; preserve the digest and the local Ollama model artifacts used for a measured run. Evaluation with `--answers` records the available model's digest as well as the session model name. Retrieval-only reports record the session answer-model name but do not attest to installed answer-model weights. Downloading the same tag later is not a guarantee of identical weights.

The embedding model name is stored in the session and evaluation report, but the current FastEmbed integration does not pin a remote model revision or hash every cached embedding artifact. Preserve `.papertrail/models` along with the saved vectors for a closer reproduction. Cached artifacts, ONNX runtime versions, hardware, and model-service versions can affect results; a fixed generation seed and temperature do not guarantee identical outputs across environments.

Preserve the source PDF, `parsed.json`, saved session/index, evaluation question file, and generated report. The evaluation records the source PDF checksum, parsed-artifact checksum, and question-file checksum to identify these inputs. The report's model names and dependency snapshot help explain a run; they do not turn the small developer-constructed evaluation into a held-out benchmark. See [evaluation.md](evaluation.md) for the scoring protocol and its limits.
