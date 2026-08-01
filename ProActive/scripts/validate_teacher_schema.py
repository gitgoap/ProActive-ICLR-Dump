#!/usr/bin/env python3
"""
Teacher cache JSONL schema validator CLI (Plan §14.1).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from proactive.audits.schema_validator import validate_file, validate_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("validate_schema")


def main():
    parser = argparse.ArgumentParser(description="Validate teacher cache JSONL schema.")
    parser.add_argument("file_path", type=str, help="Path to teacher cache JSONL file.")
    parser.add_argument("--output_report", type=str, default=None, help="Optional output report path.")
    args = parser.parse_args()

    report = validate_file(Path(args.file_path))
    logger.info(f"Validated {report['total_rows']} rows. Valid: {report['valid_rows']}, Invalid: {report['invalid_rows']}")

    if args.output_report:
        out_p = Path(args.output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved schema validation report to {out_p}")

    if not report["is_valid"]:
        logger.error(f"Schema validation FAILED with {len(report['errors'])} errors!")
        sys.exit(1)
    else:
        logger.info("Schema validation PASSED.")


if __name__ == "__main__":
    main()
