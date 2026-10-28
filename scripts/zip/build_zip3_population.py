#!/usr/bin/env python3
"""Build ``src/tide2/resources/zip3_population_2020.tsv`` from the 2020 Decennial Census.

Fetches total population (``P1_001N``, DHC file) for every ZIP Code Tabulation Area (ZCTA5), sums it by the
first three digits, and writes one row per 3-digit prefix. The HIPAA Safe Harbor ZIP rule needs exactly this
number: the population of the area formed by all ZIP codes that share the first three digits.

The Census API needs a free key (https://api.census.gov/data/key_signup.html), read from ``CENSUS_API_KEY``.
Never commit the key. Run by hand when the data should be refreshed (for example after the next census):

    CENSUS_API_KEY=... uv run python scripts/zip/build_zip3_population.py

ZIPs and prefixes are strings throughout: a prefix such as ``021`` or ``000`` must keep its leading zeros.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC
from datetime import date
from datetime import datetime
from pathlib import Path

CENSUS_URL = "https://api.census.gov/data/2020/dec/dhc"
POPULATION_VARIABLE = "P1_001N"
ZCTA_FIELD = "zip code tabulation area"
ZCTA_LENGTH = 5
THRESHOLD = 20000
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "src" / "tide2" / "resources" / "zip3_population_2020.tsv"
TIMEOUT_SECONDS = 120


def fetch_zcta_population(api_key: str) -> list[dict[str, str]]:
    """Return one ``{population, zcta}`` mapping per ZCTA from the Census API."""
    query = urllib.parse.urlencode(
        {"get": POPULATION_VARIABLE, "for": f"{ZCTA_FIELD}:*", "key": api_key}, quote_via=urllib.parse.quote
    )
    with urllib.request.urlopen(f"{CENSUS_URL}?{query}", timeout=TIMEOUT_SECONDS) as response:  # noqa: S310 # nosec B310 # constant https URL
        payload = json.load(response)
    header, rows = payload[0], payload[1:]
    # Look the columns up by name: the API returns the ZCTA as the last column.
    pop_idx, zcta_idx = header.index(POPULATION_VARIABLE), header.index(ZCTA_FIELD)
    return [{"population": row[pop_idx], "zcta": row[zcta_idx]} for row in rows]


def aggregate_by_zip3(zctas: list[dict[str, str]]) -> dict[str, int]:
    """Sum ZCTA populations by their first three digits; keys stay strings with leading zeros."""
    totals: Counter[str] = Counter()
    for item in zctas:
        zcta = item["zcta"]
        if len(zcta) != ZCTA_LENGTH or not zcta.isascii() or not zcta.isdigit():
            raise ValueError(f"Unexpected ZCTA {zcta!r}")
        totals[zcta[:3]] += int(item["population"])
    return dict(sorted(totals.items()))


def render_tsv(totals: dict[str, int], retrieved: date) -> str:
    """Render the TSV with its provenance header."""
    lines = [
        f"# source: 2020 Decennial Census DHC, {POPULATION_VARIABLE} (total population) by ZCTA5, summed by first 3 digits",
        f"# retrieved: {retrieved.isoformat()}  script: scripts/zip/build_zip3_population.py",
        "zip3\tpopulation",
        *(f"{zip3}\t{population}" for zip3, population in totals.items()),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    """Fetch, aggregate, write the TSV, and print the restricted prefixes."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="TSV path to write")
    args = parser.parse_args()

    api_key = os.environ.get("CENSUS_API_KEY", "").strip()
    if not api_key:
        print("Set CENSUS_API_KEY (free key: https://api.census.gov/data/key_signup.html)", file=sys.stderr)
        return 1

    totals = aggregate_by_zip3(fetch_zcta_population(api_key))
    args.output.write_text(render_tsv(totals, datetime.now(UTC).date()), encoding="utf-8")

    restricted = {zip3: pop for zip3, pop in totals.items() if pop <= THRESHOLD}
    print(f"{len(totals)} prefixes, total population {sum(totals.values())}, wrote {args.output}")
    print(f"{len(restricted)} restricted (<= {THRESHOLD}): {restricted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
