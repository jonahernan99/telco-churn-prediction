"""Check that every decimal number quoted in the notebook's text appears in its outputs.

The original report quoted Logistic Regression results that did not match the notebook. This script
prevents that from happening again: it reads the executed notebook, collects every number with a
decimal point from the markdown cells, and checks that each one appears in some code-cell output.

A text number matches an output number if both round to the same value at the text's precision
(so "0.62" in an output matches "0.620" in the text, and "0.6407" matches "0.641"). Numbers that are
calculated from outputs, such as a train-test gap, must be listed in DERIVED with the calculation.

Usage:
    python scripts/check_numbers.py [notebooks/churn_prediction.ipynb]

Exits with status 1 if any number is not found.
"""

from __future__ import annotations

import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

DEFAULT_NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "churn_prediction.ipynb"

# Decimal numbers, optionally with thousands separators. A number may end a sentence ("F1 was 0.614."),
# but it must not be part of a longer token such as a version ("3.11.2") or an identifier.
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)\.(\d+)(?!\w)(?!\.\d)")
# Section and version references are not results.
NOT_A_RESULT = re.compile(r"(Section|Sections|Part|Python|scikit-learn|pandas)\s*$|^#+\s*$")

# Numbers in the text that are calculated from the outputs rather than printed by them.
DERIVED = {
    "0.27": "train-validation accuracy gap with min_samples_leaf = 1: 0.998 - 0.733",
    "0.03": "train-validation accuracy gap with min_samples_leaf = 50: 0.815 - 0.789",
    "0.002": "largest train-validation accuracy gap across the values of C",
    "0.01": "train-test accuracy gaps of Models 1 and 4: 0.813 - 0.804 and 0.782 - 0.772",
    "0.02": "train-test accuracy gap of Model 3: 0.762 - 0.740",
    "0.254": "train-test accuracy gap of Model 2: 0.998 - 0.744",
    "0.010": "train-test accuracy gap of Model 4: 0.782 - 0.772",
    "2.8": "balanced class weight ratio: 4,139 non-churners / 1,495 churners in the training set",
    "4.5": "extra offers per extra churner, Model 3 vs Model 4: 81 / 18 (printed as 4.50)",
    "0.80": "lowest 'No churn' F1 across the models, from the classification reports",
    "0.87": "highest 'No churn' F1 across the models, from the classification reports",
}


def output_text(cell: dict) -> str:
    parts = []
    for output in cell.get("outputs", []):
        if "text" in output:
            parts.append("".join(output["text"]))
        data = output.get("data", {})
        for kind in ("text/plain", "text/html"):
            if kind in data:
                parts.append("".join(data[kind]))
    return "\n".join(parts)


def decimals_in(text: str) -> list[tuple[str, int]]:
    """Return (number, position) pairs for decimal numbers that look like results."""
    found = []
    for match in NUMBER.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        before = text[line_start : match.start()]
        if NOT_A_RESULT.search(before):
            continue
        found.append((match.group(0).replace(",", ""), match.start()))
    return found


def rounds_to(value: Decimal, target: Decimal) -> bool:
    exponent = target.as_tuple().exponent
    return value.quantize(Decimal(1).scaleb(exponent), rounding=ROUND_HALF_UP) == target


def check(notebook_path: Path) -> list[str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if not any(cell.get("outputs") for cell in cells if cell["cell_type"] == "code"):
        return [f"{notebook_path} has no outputs: execute it first"]

    outputs = "\n".join(output_text(cell) for cell in cells if cell["cell_type"] == "code")
    output_numbers = {Decimal(number) for number, _ in decimals_in(outputs)}
    # Percentages are printed both as 0.921 and as 92.1%; accept either form.
    output_numbers |= {number * 100 for number in output_numbers}

    problems = []
    for index, cell in enumerate(cells):
        if cell["cell_type"] != "markdown":
            continue
        text = "".join(cell["source"])
        for number, position in decimals_in(text):
            if number in DERIVED:
                continue
            target = Decimal(number)
            if any(rounds_to(value, target) for value in output_numbers):
                continue
            line = text.count("\n", 0, position) + 1
            problems.append(f"cell {index}, line {line}: {number} does not appear in any output")
    return problems


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT_NOTEBOOK
    problems = check(path)
    for problem in problems:
        print(problem)
    if problems:
        print(f"\n{len(problems)} number(s) in the text are not backed by an output.")
        return 1
    print("Every decimal number in the text matches an output or a documented calculation.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
