"""
Serialization utilities for Presidio analyzer and anonymizer results.

This module provides functions to serialize and deserialize various Presidio result objects
to/from JSON format, enabling persistence and reloading of analysis and anonymization results.

Key Functions:
- dict_analyzer_result_to_dict/json: Serialize DictAnalyzerResult objects
- json_to_dict_analyzer_result: Deserialize JSON to DictAnalyzerResult objects
- engine_result_to_dict: Convert EngineResult to JSON serializable format
- dict_to_engine_result: Convert JSON dict back to EngineResult object
- json_to_engine_result: Load EngineResult objects from JSON files/directories

Example usage for EngineResult:
    # Load anonymized results from JSON
    from tide2.utils.serialization import json_to_engine_result

    # Load from a single file
    results = list(json_to_engine_result('path/to/result.json'))

    # Load from a directory of JSON files
    results = list(json_to_engine_result('path/to/results_directory/'))

    # Process the loaded results
    for result in results:
        print(f"Anonymized text: {result.text}")
        for item in result.items:
            print(f"  {item.entity_type}: {item.text}")
"""

import glob
import json
import os
from collections.abc import Iterator

from presidio_analyzer.dict_analyzer_result import DictAnalyzerResult
from presidio_analyzer.recognizer_result import RecognizerResult
from presidio_anonymizer.entities import EngineResult
from presidio_anonymizer.entities.engine.result.operator_result import OperatorResult


def dict_analyzer_result_to_dict(result: DictAnalyzerResult) -> dict:
    """
    Convert DictAnalyzerResult to JSON serializable format.

    Args:
        result: A Presidio DictAnalyzerResult to convert.

    Returns:
        Dict with keys "key", "value", and "recognizer_results" (list of dicts).
    """
    return {
        "key": result.key,
        "value": result.value,
        "recognizer_results": [x.to_dict() for x in result.recognizer_results],
    }


def dict_to_dict_analyzer_result(json_dict: dict) -> DictAnalyzerResult:
    """
    Convert a dictionary to DictAnalyzerResult.

    Args:
        json_dict: Dict with keys "key", "value", and "recognizer_results".

    Returns:
        A reconstructed DictAnalyzerResult instance.
    """
    return DictAnalyzerResult(
        key=json_dict["key"],
        value=json_dict["value"],
        recognizer_results=[RecognizerResult.from_json(x) for x in json_dict["recognizer_results"]],
    )


def dict_analyzer_result_to_json(result: DictAnalyzerResult | Iterator[DictAnalyzerResult], output_folder: str) -> None:
    """
    Serialize DictAnalyzerResult(s) to JSON files.

    Each result is written to a separate JSON file named by its key.

    Args:
        result: A single DictAnalyzerResult or a list of them.
        output_folder: Directory path where JSON files will be written.
            Created if it does not exist.
    """

    if not isinstance(result, list):
        result_list = [result]
    else:
        result_list = result

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for r in result_list:
        if not isinstance(r, DictAnalyzerResult):
            raise TypeError("Expected DictAnalyzerResult or list of DictAnalyzerResult")

        json_dict = dict_analyzer_result_to_dict(r)
        with open(f"{output_folder}/{json_dict['key']}.json", "w") as f:
            json.dump(json_dict, f, indent=4)


def json_to_dict_analyzer_result(fname: str) -> Iterator[DictAnalyzerResult]:
    """
    Deserialize a JSON file or a folder with JSON files to a list of DictAnalyzerResult.
    """
    if not os.path.isdir(fname):
        if not fname.endswith(".json"):
            raise ValueError("Input must be a JSON file or a directory containing JSON files.")
        file_list = [fname]
    else:
        file_list = glob.glob(os.path.join(fname, "*.json"))
        if not file_list:
            raise ValueError("No JSON files found in the specified directory.")

    for file in file_list:
        with open(os.path.join(fname, file)) as f:
            json_dict = json.load(f)
            yield dict_to_dict_analyzer_result(json_dict)


def serialize_dict_analyzer_results(analyzer_results: dict, fpath: str) -> None:
    """
    Serialize analyzer results to a single JSON file.

    Args:
        analyzer_results: Iterable of DictAnalyzerResult objects. Each result's
            key maps to its first recognizer result list serialized as dicts.
        fpath: Output file path for the JSON file.
    """
    json_dict = {rst.key: [ent.to_dict() for ent in rst.recognizer_results[0]] for rst in analyzer_results}

    with open(fpath, "w") as f:
        json.dump(json_dict, f, indent=4)


def engine_result_to_dict(result: EngineResult) -> dict:
    """
    Convert EngineResult to JSON serializable format.

    Args:
        result: A Presidio EngineResult to convert.

    Returns:
        Dict with keys "text" and "items" (list of operator result dicts).
    """
    return {"text": result.text, "items": [item.__dict__ for item in result.items] if result.items else []}


def dict_to_engine_result(json_dict: dict) -> EngineResult:
    """
    Convert a dictionary to EngineResult.

    Args:
        json_dict: Dict with keys "text" and optionally "items".

    Returns:
        A reconstructed EngineResult instance.
    """
    items = []
    if json_dict.get("items"):
        items = [OperatorResult.from_json(item) for item in json_dict["items"]]

    return EngineResult(text=json_dict.get("text", ""), items=items)


def engine_result_to_json(result: EngineResult | Iterator[EngineResult], output_folder: str) -> None:
    """
    Serialize EngineResult(s) to JSON file(s).

    Each result is written to a separate file named ``engine_result_{i}.json``.

    Args:
        result: A single EngineResult or a list of them.
        output_folder: Directory path where JSON files will be written.
            Created if it does not exist.
    """
    if not isinstance(result, list):
        result_list = [result]
    else:
        result_list = result

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    for i, r in enumerate(result_list):
        if not isinstance(r, EngineResult):
            raise TypeError("Expected EngineResult or list of EngineResult")

        json_dict = engine_result_to_dict(r)
        # Use index as filename if no other identifier is available
        filename = f"engine_result_{i}.json"
        with open(f"{output_folder}/{filename}", "w") as f:
            json.dump(json_dict, f, indent=4)


def json_to_engine_result(fname: str) -> Iterator[EngineResult]:
    """
    Deserialize a JSON file or a folder with JSON files to EngineResult objects.

    Args:
        fname: Path to a JSON file or directory containing JSON files

    Yields:
        EngineResult objects reconstructed from JSON
    """
    if not os.path.isdir(fname):
        if not fname.endswith(".json"):
            raise ValueError("Input must be a JSON file or a directory containing JSON files.")
        file_list = [fname]
    else:
        file_list = glob.glob(os.path.join(fname, "*.json"))
        if not file_list:
            raise ValueError("No JSON files found in the specified directory.")

    for file in file_list:
        file_path = file if not os.path.isdir(fname) else os.path.join(fname, os.path.basename(file))
        with open(file_path) as f:
            json_dict = json.load(f)
            yield dict_to_engine_result(json_dict)
