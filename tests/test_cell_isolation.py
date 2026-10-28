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

ok = cell.anonymize_cell("48291037", "MRN", b"\\x01" * 32, b"\\x02" * 32)
modules = sorted(sys.modules)  # before any ZIP call: the ZIP module must load lazily
zip_out = cell.anonymize_cell("02139", "ZIP_CODE", b"\\x01" * 32, b"\\x02" * 32, zip_mode="safe-harbor-strict")
print(json.dumps({"ok": ok, "modules": modules, "zip": zip_out}))
"""


def test_cell_loads_without_package_roots_or_heavy_modules() -> None:
    proc = subprocess.run([sys.executable, "-c", CHILD, str(SRC)], capture_output=True, text=True, check=False)  # noqa: S603
    assert proc.returncode == 0, proc.stderr[-2000:]
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result["ok"]
    assert result["zip"] == "021"
    loaded = result["modules"]
    assert "tide2.anonymizers.zip_code" not in loaded
    for forbidden in FORBIDDEN:
        assert not any(m == forbidden or m.startswith(forbidden + ".") for m in loaded), forbidden
