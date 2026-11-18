import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import check_numbers  # noqa: E402


def make_notebook(tmp_path, markdown, outputs):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [text]} for text in markdown]
    cells += [
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 1,
            "source": ["print()"],
            "outputs": [{"output_type": "stream", "name": "stdout", "text": [text]}],
        }
        for text in outputs
    ]
    path = tmp_path / "notebook.ipynb"
    path.write_text(json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}))
    return path


def test_matching_numbers_pass(tmp_path):
    path = make_notebook(tmp_path, ["Recall is **0.778** and F1 0.614."], ["recall 0.778\nf1 0.614"])
    assert check_numbers.check(path) == []


def test_number_missing_from_outputs_is_reported(tmp_path):
    path = make_notebook(tmp_path, ["Recall is 0.781."], ["recall 0.778"])
    problems = check_numbers.check(path)
    assert len(problems) == 1
    assert "0.781" in problems[0]


def test_number_at_the_end_of_a_sentence_is_checked(tmp_path):
    path = make_notebook(tmp_path, ["Recall is 0.781.\nF1 was 0.617."], ["recall 0.778"])
    assert len(check_numbers.check(path)) == 2


def test_version_strings_are_not_numbers(tmp_path):
    path = make_notebook(tmp_path, ["Built with matplotlib 3.11.2."], ["nothing here"])
    assert check_numbers.check(path) == []


def test_rounding_to_the_text_precision(tmp_path):
    # pandas prints 0.620 as 0.62, and longer values are rounded in the text.
    path = make_notebook(tmp_path, ["F1 0.620, AUC 0.641, k > 4.50"], ["0.62 0.6407 4.5"])
    assert check_numbers.check(path) == []


def test_rounding_does_not_accept_a_different_value(tmp_path):
    path = make_notebook(tmp_path, ["F1 0.642"], ["0.6407"])
    assert len(check_numbers.check(path)) == 1


def test_percentages_match_fractions(tmp_path):
    path = make_notebook(tmp_path, ["The rule agrees on 92.1% of customers."], ["fidelity 0.921"])
    assert check_numbers.check(path) == []


def test_section_references_are_ignored(tmp_path):
    path = make_notebook(tmp_path, ["## 5.4 Costs\nSee Section 5.6 and Part 2.7."], ["nothing here"])
    assert check_numbers.check(path) == []


def test_thousands_separator(tmp_path):
    path = make_notebook(tmp_path, ["Total charges up to 8,684.80."], ["8684.8"])
    assert check_numbers.check(path) == []


def test_derived_numbers_are_allowed(tmp_path):
    gap = next(iter(check_numbers.DERIVED))
    path = make_notebook(tmp_path, [f"The gap is {gap}."], ["nothing here"])
    assert check_numbers.check(path) == []


def test_unexecuted_notebook_is_an_error(tmp_path):
    path = make_notebook(tmp_path, ["Recall 0.778"], [])
    problems = check_numbers.check(path)
    assert problems and "execute it first" in problems[0]


@pytest.mark.skipif(not check_numbers.DEFAULT_NOTEBOOK.exists(), reason="notebook not found")
def test_project_notebook_is_consistent():
    assert check_numbers.check(check_numbers.DEFAULT_NOTEBOOK) == []
