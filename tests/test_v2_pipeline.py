"""Test suite to move from original pipeline.py script to pipeline_v2.py

Confidence is high in the function of our initial pipeline.
Beginning refactor with the goal of better code quality.
"""

import pytest
import pandas as pd

from scripts.pipeline import process_data as initial_process_data
from scripts.pipeline_v2 import process_data as v2_process_data
from cgfp.util import load_to_pd


def test_imports():
    assert True


test_args = [
    # (input_file, raw_data, sample)  # (name of input file with no path, path to folder or test folder acting as raw data source, whether to sample)
    ("test_input_1.csv", "./tests/assets/", False),
    ("test_input_2.csv", "./tests/assets/", False),
    # ("CONFIDENTIAL_CGFP bulk data_073123.xlsx", "./data/raw/", True),
    ("CONFIDENTIAL_CGFP bulk data_073123.xlsx", "./data/raw/", False),
]


@pytest.mark.parametrize("input_file, raw_data, sample", test_args)
def test_initial_and_v2_pipelines_match(input_file, raw_data, sample, v2_arg_parser):
    """Setting this up to be parametrized

    NOTE: It would be great if we could turn off printing on the initial pipeline version.
    Right now it will put outputs under the ./data/clean/RUN folder while testing, same as with the usual run

    See: https://docs.pytest.org/en/7.1.x/example/parametrize.html
    """
    data = load_to_pd(raw_data, input_file)

    if sample:
        data = data.sample(n=50, random_state=1)

    argv = ["--input_file", input_file, "--raw_data", raw_data]

    # I don't think args are used after initial setup
    # if they are needed, extract create parser to it's own function as in v2 pipeline and repeat below process
    initial_misc, initial_output = initial_process_data(data)

    v2_options = vars(v2_arg_parser.parse_args(argv))
    v2_misc, v2_output = v2_process_data(data, **v2_options)

    mismatch_output_columns = set(v2_output.columns) ^ set(initial_output.columns)
    assert len(mismatch_output_columns) == 0, "Output columns are mismatching"
    # TODO: Can I output the mismatched rows?
    assert v2_output.equals(
        initial_output
    ), "Output columns seem to match but other issue in dataframe"

    mismatch_misc_columns = set(v2_misc.columns) ^ set(initial_misc.columns)
    assert len(mismatch_misc_columns) == 0, "Misc columns are mismatching"
    assert v2_misc.equals(
        initial_misc
    ), "Misc columns seem to match but other issue in dataframe"
