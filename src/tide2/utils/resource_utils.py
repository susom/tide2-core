"""
Resource utilities for accessing package resource files.

This module provides utilities for accessing resource files bundled with
the tide2 package, regardless of installation method.
"""

import hashlib
import importlib.resources as pkg_resources
import json
from pathlib import Path


def get_resource_path(filename: str) -> str:
    """
    Get the absolute path to a resource file in the package.

    Args:
        filename: Name of the resource file (e.g., 'first_names_with_sex.tsv')

    Returns:
        Absolute path to the resource file

    Raises:
        FileNotFoundError: If the resource file doesn't exist
    """
    # Get the directory containing this module
    current_dir = Path(__file__).parent

    # Navigate to the resources directory (go up one level from utils to tide2, then to resources)
    resources_dir = current_dir.parent / "resources"

    # Construct the full path to the resource file
    resource_path = resources_dir / filename

    if not resource_path.exists():
        raise FileNotFoundError(f"Resource file '{filename}' not found at {resource_path}")

    return str(resource_path.absolute())


def list_resources() -> list[str]:
    """
    List all available resource files.

    Returns:
        List of resource filenames
    """
    current_dir = Path(__file__).parent
    resources_dir = current_dir.parent / "resources"

    if not resources_dir.exists():
        return []

    return [f.name for f in resources_dir.iterdir() if f.is_file()]


# Common resource file constants
FIRST_NAMES_FILE = "first_names_with_sex.tsv"
SURNAMES_FILE = "surnames.tsv"
UNIFIED_NAMES_FILE = "unified_names.tsv"
STREET_NAMES_FILE = "street_names.tsv"
ZIPCODES_FILE = "zipcodes.tsv"
CITIES_FILE = "cities.tsv"
STATES_FILE = "states.tsv"
COUNTRIES_FILE = "countries.tsv"
BERT_TRANSFORMER_CONFIG_FILE = "bert_transformer_configuration.json"
STOPWORDS_FILE = "stopwords.txt"
HOSPITALS_FILE = "hospitals.txt"
LLM_PROMPTS_DIR = "llm_prompts"


def md5sum(filename):
    """Calculate the MD5 checksum of a file."""
    md5 = hashlib.md5()  # noqa: S324 # nosec B324
    with Path(filename).open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


def load_llm_prompt(prompt_name: str) -> dict:
    """
    Load an LLM prompt config and template.

    When *prompt_name* is a plain name (e.g. ``"phi_detection"``), the config
    and template are loaded from the bundled ``resources/llm_prompts/`` directory.

    When *prompt_name* is a directory path (e.g. ``"/my/prompts/custom/"``),
    the config is loaded from ``<dir>/<basename>.json`` and the template from
    the ``prompt_file`` key inside it, resolved relative to the same directory.

    Args:
        prompt_name: Either a bundled prompt name (without extension) or a
            directory path containing ``<basename>.json`` and its template file.

    Returns:
        The loaded prompt configuration dict, including all keys from the JSON
        config file, with an added ``prompt_template`` key containing the
        template contents as a string.

    Raises:
        FileNotFoundError: If config or template file not found
        ValueError: If prompt_name contains invalid characters or config is malformed
    """
    prompt_path = Path(prompt_name)

    if prompt_path.is_dir():
        # External directory: derive config name from the directory basename
        prompts_dir = prompt_path
        basename = prompt_path.name
        config_path = prompts_dir / f"{basename}.json"
        return _load_external_prompt(prompt_name, prompts_dir, config_path)

    # Bundled resource name: validate to prevent path traversal
    if "/" in prompt_name or "\\" in prompt_name or prompt_name.startswith("."):
        raise ValueError(
            f"Invalid prompt name '{prompt_name}': bundled prompt names cannot contain "
            "path separators or start with '.'. Use a directory path for external prompts."
        )
    return _load_bundled_prompt(prompt_name)


def _load_bundled_prompt(prompt_name: str) -> dict:
    """Load a prompt config and template from bundled package resources."""
    from tide2.resources import llm_prompts as llm_prompts_pkg

    config_filename = f"{prompt_name}.json"
    prompts_ref = pkg_resources.files(llm_prompts_pkg)
    config_ref = prompts_ref.joinpath(config_filename)

    try:
        config_text = config_ref.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"LLM prompt config '{prompt_name}' not found in bundled resources") from None

    config = json.loads(config_text)
    _validate_prompt_config(prompt_name, config)

    template_ref = prompts_ref.joinpath(config["prompt_file"])
    try:
        config["prompt_template"] = template_ref.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"LLM prompt template '{config['prompt_file']}' not found in bundled resources"
        ) from None

    return config


def _load_external_prompt(prompt_name: str, prompts_dir: Path, config_path: Path) -> dict:
    """Load a prompt config and template from an external directory."""
    if not config_path.exists():
        raise FileNotFoundError(f"LLM prompt config '{prompt_name}' not found at {config_path}")

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    _validate_prompt_config(prompt_name, config)

    prompt_file = config["prompt_file"]
    prompt_file_path = Path(prompt_file)
    if prompt_file_path.is_absolute():
        raise ValueError(f"LLM prompt config '{prompt_name}' has invalid prompt_file: absolute paths are not allowed")

    resolved_prompts_dir = prompts_dir.resolve()
    template_path = (prompts_dir / prompt_file_path).resolve()
    try:
        template_path.relative_to(resolved_prompts_dir)
    except ValueError as exc:
        raise ValueError(
            f"LLM prompt config '{prompt_name}' has invalid prompt_file: path escapes prompt directory"
        ) from exc
    if not template_path.exists():
        raise FileNotFoundError(f"LLM prompt template not found at {template_path}")

    with template_path.open(encoding="utf-8") as f:
        config["prompt_template"] = f.read()

    return config


def _validate_prompt_config(prompt_name: str, config: dict) -> None:
    """Validate required keys in a prompt config dict."""
    required_keys = ["prompt_file", "supported_entities"]
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"LLM prompt config '{prompt_name}' is missing required keys: {missing_keys}")

    prompt_file = config["prompt_file"]
    if not isinstance(prompt_file, str) or not prompt_file:
        raise ValueError(f"LLM prompt config '{prompt_name}' has invalid prompt_file: expected non-empty string")

    supported_entities = config["supported_entities"]
    if not isinstance(supported_entities, list) or not supported_entities:
        raise ValueError(f"LLM prompt config '{prompt_name}' has invalid supported_entities: expected a non-empty list")


def load_stopwords() -> frozenset[str]:
    """
    Load stopwords from the resource file.

    Loads stopwords from the stopwords.txt resource file. If the stopwords
    file cannot be loaded, returns an empty set and prints a warning.

    Note: This function does not implement caching. For thread-safe caching,
    see the KnownValuesRecognizer class which implements double-checked locking.

    Returns:
        Frozenset of stopwords in lowercase for case-insensitive matching.
        Returns empty frozenset if stopwords file cannot be loaded.
    """
    try:
        stopwords_path = Path(get_resource_path(STOPWORDS_FILE))
        with stopwords_path.open(encoding="utf-8") as f:
            return frozenset(word.strip().lower() for word in f.readlines() if word.strip())
    except (FileNotFoundError, OSError) as e:
        print(f"Warning: Could not load stopwords file: {e}")
        return frozenset()
