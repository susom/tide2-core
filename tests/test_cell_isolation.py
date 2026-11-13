"""The cell module must load with only the package roots missing, as in the BigQuery bundle."""

import json
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "tide2"
FORBIDDEN = ("faker", "usaddress", "tide2.actors", "tide2.string_parsers", "tide2.anonymizers.faker")

CHILD = """
import json, sys, types

# Namespace packages: neither tide2/__init__.py nor tide2/anonymizers/__init__.py is executed.
for name, sub in (("tide2", ""), ("tide2.anonymizers", "anonymizers")):
    pkg = types.ModuleType(name)
    pkg.__path__ = [sys.argv[1] + ("/" + sub if sub else "")]
    sys.modules[name] = pkg

import tide2.anonymizers.cell as cell

print(json.dumps({"ok": cell.anonymize_cell("48291037", "MRN", b"\\x01" * 32, b"\\x02" * 32), "modules": sorted(sys.modules)}))
"""


def test_cell_loads_without_package_roots_or_heavy_modules() -> None:
    proc = subprocess.run([sys.executable, "-c", CHILD, str(SRC)], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["ok"]
    loaded = result["modules"]
    for forbidden in FORBIDDEN:
        assert not any(m == forbidden or m.startswith(forbidden + ".") for m in loaded), forbidden


BLOCKED = ("presidio_analyzer", "spacy", "transformers", "torch", "sklearn", "scipy", "pandas", "pyarrow", "ray")

FREE_TEXT_CHILD = """
import importlib.abc, json, sys, types

BLOCKED = tuple(sys.argv[2].split(","))
ROOT = sys.argv[1]
attempts = []


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] not in BLOCKED:
            return None
        frame = sys._getframe(1)
        while frame is not None and "importlib" in frame.f_code.co_filename:
            frame = frame.f_back
        attempts.append([name, frame.f_code.co_filename if frame is not None else ""])
        raise ImportError(f"blocked: {name}")


sys.meta_path.insert(0, Blocker())

# Only the two package roots are stubbed; tide2.actors and tide2.utils run their real initializers.
for name, sub in (("tide2", ""), ("tide2.anonymizers", "anonymizers")):
    pkg = types.ModuleType(name)
    pkg.__path__ = [ROOT + ("/" + sub if sub else "")]
    sys.modules[name] = pkg

import tide2.anonymizers.cell as cell

text = "Patient Rhea Orleans, MRN 48291037."
spans = [
    {"entity_type": "PERSON", "start": 8, "end": 20, "score": 1.0},
    {"entity_type": "MRN", "start": 26, "end": 34, "score": 1.0},
]
out = cell.anonymize_cell(text, "FREE_TEXT", b"\\x01" * 32, b"\\x02" * 32, "P1", recognizer_results=spans)
print(json.dumps({"out": out, "modules": sorted(sys.modules), "attempts": attempts}))
"""


def test_free_text_loads_without_heavy_dependencies() -> None:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", FREE_TEXT_CHILD, str(SRC), ",".join(BLOCKED)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "Rhea" not in result["out"]
    assert "48291037" not in result["out"]
    loaded = result["modules"]
    for name in (*BLOCKED, "tide2.actors.recognizer", "tide2.actors.transformer"):
        assert not any(m == name or m.startswith(name + ".") for m in loaded), name
    # An ImportError caught by an optional-import guard in tide2 code would pass the checks above.
    tide2_attempts = [a for a in result["attempts"] if a[1].startswith(str(SRC))]
    assert not tide2_attempts, tide2_attempts
