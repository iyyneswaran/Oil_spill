"""Create JSON and GeoJSON hindcast output from supplied historical forcing.

Usage:
    uv run python scripts/hindcast.py --input hindcast-input.json --output-dir artifacts/hindcast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oilspill.hindcast import HindcastInput, run_hindcast


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtrack an oil slick with historical currents and wind."
    )
    parser.add_argument("--input", type=Path, required=True, help="Hindcast input JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for hindcast.json and hindcast.geojson",
    )
    args = parser.parse_args()

    request = HindcastInput.model_validate_json(args.input.read_text(encoding="utf-8"))
    result = run_hindcast(request)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hindcast.json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "hindcast.geojson").write_text(
        json.dumps(result.to_geojson(), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
