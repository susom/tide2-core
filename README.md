# TIDE 2.0

A data de-identification and anonymization toolkit that combines multiple anonymization strategies with cryptographic techniques and machine learning-based entity recognition.

## Get Started

### Dev Container (recommended)

The repository includes a [Dev Container](https://containers.dev/) configuration that sets up
the full development environment automatically: Python 3.12, `uv`, all dependencies (including
GPU group), pre-commit hooks.

**Prerequisites:**
- [Docker](https://docs.docker.com/get-docker/)
- [VS Code](https://code.visualstudio.com/) with the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

**Steps:**

1. Clone the repository and open it in VS Code:
   ```bash
   git clone https://github.com/susom/tide2-core.git
   cd tide2
   ```
2. When VS Code detects `.devcontainer/devcontainer.json`, click **Reopen in Container**
   (or run the command **Dev Containers: Reopen in Container** from the command palette).
3. The virtual environment at `/opt/tide2-core/.venv` is activated by default in all terminals.

The Dev Container includes these VS Code extensions pre-installed: Python, Ruff, Jupyter,
Docker, and TOML support.

### Local Installation (without Dev Container)

If you prefer to develop outside the Dev Container:

```bash
# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/susom/tide2.git
cd tide2

uv python install 3.12.8
uv sync

# Activate the virtual environment before running any Python commands
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows (PowerShell)
```

### Quick Start: Interactive Tutorial

The tutorial notebook walks you through the de-identification pipeline step by step.

**In the Dev Container (or local VS Code):**

1. Open `notebooks/tide2_pipeline.ipynb` (the Jupyter extension is pre-installed in the Dev Container)
2. When prompted for a kernel, select the `.venv (Python 3.12)` environment

**Jupyter in the browser (local installation):**

```bash
uv sync --group dev              # Install Jupyter (dev dependency group)
source .venv/bin/activate
jupyter notebook notebooks/tide2_pipeline.ipynb
```

View the notebook on GitHub: [TIDE 2.0 Pipeline Tutorial](https://github.com/susom/tide2/blob/main/notebooks/tide2_pipeline.ipynb)

**Troubleshooting:**
- **Run from the repo root** — launch Jupyter from the `tide2/` directory so that relative paths resolve correctly.
- **No cloud credentials required** — the notebook downloads the transformer model from HuggingFace Hub by default.
- **Kernel crashes** — if the Jupyter kernel crashes repeatedly, restart Jupyter (`Ctrl+C`, then re-launch) and run the cells from the top.

---

## Overview

TIDE 2.0 is a Python package for anonymizing sensitive data in healthcare and research contexts. It identifies and anonymizes personally identifiable information (PII) while maintaining data utility for analysis and research.

## Features

### Entity Recognition
- **Transformer-based NER**: HuggingFace transformer models with direct batch inference (bypasses HF pipeline), BIO token aggregation, and chunk-to-document reassembly
- **Regex recognizers**: Phone, URL/IP, Email, SSN, Address — replacements for Presidio defaults (10-100x faster)
- **Healthcare-specific**: MRN, Accession Number, HAR code recognizers
- **Known values detection**: Aho-Corasick based matching against patient databases
- **Specialized**: Base64 image detection, genetic sequence detection, LLM-based JSON recognizer
- **Cached results**: Pre-computed NER results from GPU batch processing via `CachedResultsTransformerRecognizer`
- **Presidio Integration**: Built on Microsoft's Presidio framework

### Anonymization Strategies
- **HIPS (Healthcare Identity Protection System)**: Cryptographic deterministic anonymization for names, locations, and alphanumeric identifiers
- **Accession number hashing**: SHA256-based, compatible with BigQuery UDF
- **Faker Integration**: Realistic fake data generation
- **Date Jittering**: Deterministic, privacy-preserving date shifts derived from patient keys
- **Age Grouping**: Age range categorization

### Cryptographic Protection
- **Format-Preserving Encryption (FPE)**: Maintains data format during encryption
- **Key Management**: Key generation, storage, and derivation utilities
- **Deterministic date jitter**: Batch-capable date shift derivation from cryptographic keys
- **String Selection**: HMAC-based cached string selection

### Transformer NER Inference
- **Direct inference**: Bypasses HuggingFace pipeline dispatch loop with batch tokenize → single GPU forward pass → offset-based extraction
- **Adaptive GPU batching**: Auto-computes batch size from model config and free GPU memory; adjusts based on text lengths with VRAM-aware budgets
- **OOM recovery**: Automatic batch splitting on CUDA out-of-memory errors
- **Chunking + reassembly**: Long documents are split into overlapping chunks for inference and reassembled into document-level entities

### Utilities
- **Text processing**: Text chunking, BIO aggregation, span reconstruction, deduplication
- **String parsers**: Name parsing/classification, address parsing, format detection
- **Span metrics**: Gold vs ML evaluation, O(n log n) conflict resolution
- **Model cache**: Auto-download transformer models from HuggingFace Hub to `~/.cache/tide2/`
- **Model compilation**: `torch.compile` with mega-cache support for faster inference startup

## Usage

TIDE 2.0 is a library — there is no bundled CLI or batch runner. Wire up a
Presidio `AnalyzerEngine` with the recognizers you need and an
`AnonymizerEngine` with the anonymizers you need, then call them directly:

```python
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from tide2.anonymizers import HipsNamesAnonymizer
from tide2.recognizers import PhoneRecognizer, TransformersRecognizer

registry = RecognizerRegistry()
registry.add_recognizer(TransformersRecognizer(model_name="StanfordAIMI/stanford-deidentifier-v2"))
registry.add_recognizer(PhoneRecognizer())
analyzer = AnalyzerEngine(registry=registry)

anonymizer = AnonymizerEngine()
anonymizer.add_anonymizer(HipsNamesAnonymizer)

results = analyzer.analyze(text="Call Dr. Smith at 555-123-4567.", language="en")
anonymized = anonymizer.anonymize(
    text="Call Dr. Smith at 555-123-4567.",
    analyzer_results=results,
    operators={"PERSON": OperatorConfig("hips_names", {"salt": salt, "key": key})},
)
```

See [`notebooks/tide2_pipeline.ipynb`](notebooks/tide2_pipeline.ipynb) for a
complete, runnable end-to-end example (regex + transformer recognizers, known
patient values, and the full HIPS anonymizer operator set).

## Docker Images

Several targets are built from a single multi-stage `Dockerfile`:

- `production-cpu` — slim CPU-only image (no CUDA). Used for recognizer/anonymizer workloads.
- `production-gpu` — GPU image based on `nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04`. Used for transformer inference. (The ML stack — `torch`, `transformers`, `spacy` — ships in both images, since it is a required core dependency.)
- `development` — Dev Container target with `git`, `gcloud`, build tools, and the full dev environment.
- `test` — extends `development` and runs the test suite (used by `make test-docker`).

Build and push the GPU image (requires `DOCKER_REGISTRY` and `DOCKER_IMAGE_GPU` in `.env`):

```bash
make docker         # build + push the GPU image (alias for docker-gpu)
make docker-gpu     # build + push the GPU image
make test-docker    # build the test target and run the suite in Docker
```

## Dependency Groups

- **`llm`**: LLM provider SDKs for the optional LLM-based recognizer (`anthropic`, `openai`, `google-genai`, `google-cloud-aiplatform`)
- **`dev`**: Development tools (`pytest`, `pytest-cov`, `ty`, `ruff`, `pre-commit`), Jupyter, and the `evaluation` libraries (`scikit-learn`, `scipy`, `tqdm`)
- **`evaluation`**: Evaluation/analysis libraries (`scikit-learn`, `scipy`, `tqdm`)
- **`test`**: Minimal test dependencies (`pytest`, `pytest-cov`)
- **`docs`**: API documentation generation (`pdoc`)

Install an optional group as an extra with `uv sync --extra <name>`, or all extras with `uv sync --all-extras`. (These same sets are also defined as `[dependency-groups]`, usable with `uv sync --group <name>`.)

Note: The full ML inference stack (`torch`, `transformers`, `spacy`) ships in the main package by default — it is required, since no model can run without it. The `llm` extra is only needed for the optional LLM-based recognizer.

## Architecture

```
tide2/
├── recognizers/              # PII detection (Presidio EntityRecognizer subclasses)
│   └── nlp_engine.py         # Blank-spaCy NlpEngine for regex-only recognition
├── anonymizers/              # PII replacement (Presidio Operator subclasses)
├── transformers/             # Core NER inference engine (TransformerCore)
│   ├── core.py              # Model loading, direct inference, BIO aggregation
│   ├── config.py            # Model configuration management
│   └── reassembly.py        # Chunk-to-document reassembly
├── cryptographic/            # FPE, key management, date jitter derivation
├── string_parsers/           # Name/address parsing, format detection
├── utils/
│   ├── span_metrics.py         # Evaluation metrics and conflict resolution
│   ├── text_processing.py      # Chunking, BIO aggregation, span reconstruction
│   ├── serialization.py        # RecognizerResult <-> dict conversions
│   ├── llm_model.py            # LLM client utilities
│   ├── constants.py            # Shared constants
│   └── resource_utils.py       # Resource path helpers
└── resources/                # Config files (model configs, name lists, etc.)
```

## Testing

```bash
# Run all unit tests (coverage report prints automatically)
uv run pytest

# Run without coverage (faster, useful when debugging)
uv run pytest --no-cov

# Run a specific test file
uv run pytest tests/test_masking_anonymizer.py

# Skip slow integration tests
uv run pytest -m "not integration"
```

Coverage is configured in `pyproject.toml` and runs automatically with `pytest`. Three reports are generated on each run:

- **Terminal**: line-by-line missing coverage printed to stdout
- **HTML**: detailed report at `htmlcov/index.html`
- **XML**: `coverage.xml` (Cobertura format)

## Documentation

### API Reference

API documentation is hosted via GitHub Pages: [https://susom.github.io/tide2-core/](https://susom.github.io/tide2-core/)

To build or preview docs locally (generated with [pdoc](https://pdoc.dev/)):

```bash
# Install docs dependencies. pdoc imports every module (including the LLM
# utilities), so the `llm` extra is required in addition to `docs`. The ML
# stack (torch/transformers/spacy) ships in the base install.
uv sync --extra docs --extra llm

# Live preview (opens a local server with hot reload)
make docs-serve

# Generate static HTML to docs/
make docs
```

Deployment to GitHub Pages is automated: the [`.github/workflows/docs.yml`](.github/workflows/docs.yml)
workflow runs `make docs` on every push to `main` and publishes the `docs/` directory as a
Pages artifact.

### Other Resources

- **Examples**: Check the `notebooks/` directory for usage examples
- **Tests**: Test suite in `tests/` directory

## Requirements

- **Dev Container**: Recommended — provides the full environment with no manual setup (requires Docker and VS Code with the Dev Containers extension)
- **Python**: 3.12 or 3.13 (required, `>=3.12,<3.14`) — 3.14 is excluded until the `spacy`/`thinc` C-extension stack and other pinned dependencies are tested against cp314 wheels.
- **Package Manager**: uv (not pip or poetry)
- **Virtual Environment**: `.venv/` (activated automatically in the Dev Container; must be activated manually for local installs)
- **Core Dependencies**: Presidio, Cryptography, Faker, and the ML inference stack (`torch`, `transformers>=5.0`, `spacy`) — all required and shipped in the base install

## Security Considerations

- Cryptographic operations use standard libraries (cryptography, pyca/cryptography)
- Format-preserving encryption maintains data format during encryption
- Key management supports generation, storage, and rotation
- Anonymization strategies are designed to prevent re-identification

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow, branching
model, commit-message conventions, and the pull request checklist. The
[Dev Container setup](#dev-container-recommended) above provisions the full
development environment (including pre-commit hooks) automatically.

## License

This project is licensed under the MIT License - see the [LICENSE-MIT](LICENSE-MIT) file for details.

## Citation

If you use TIDE 2.0 in your research, please cite:

```bibtex
@software{tide2,
  title={TIDE 2.0: Data De-identification and Anonymization Toolkit},
  author={TIDE 2.0 Team},
  year={2025},
  url={https://github.com/susom/tide2}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/susom/tide2/issues)
- **Discussions**: [GitHub Discussions](https://github.com/susom/tide2/discussions)
- **Development**: See the Contributing section above

---

**Synthetic Data Notice**: All sample data included in this repository (under `notebooks/sample_data/`) is entirely synthetic and fabricated. No real patient data is included. See [`notebooks/sample_data/README.md`](notebooks/sample_data/README.md) for details.

**Note**: This toolkit is designed for research and development purposes. Please ensure compliance with relevant privacy laws and regulations (HIPAA, GDPR, etc.) when using in production environments.
