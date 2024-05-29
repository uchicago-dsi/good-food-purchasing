"""Test suite to move from original pipeline.py script to pipeline_v2.py

Confidence is high in the function of our initial pipeline.
Beginning refactor with the goal of better code quality.
"""

import pytest
import pandas as pd

# TODO: Move these to the package
# from scripts.pipeline import process_data as initial_process_data
from scripts.pipeline_v6 import process_data as initial_process_data
from scripts.pipeline_v7 import process_data as new_process_data
from cgfp.util import load_to_pd


def test_imports():
    assert True


test_args = [
    # (input_file, raw_data, sample)  # (name of input file with no path, path to folder or test folder acting as raw data source, whether to sample)
    ("test_input_1.csv", "./tests/assets/", False),
    ("test_input_2.csv", "./tests/assets/", False),
    ("CONFIDENTIAL_CGFP bulk data_073123.xlsx", "./data/raw/", True),
    # ("CONFIDENTIAL_CGFP bulk data_073123.xlsx", "./data/raw/", False),
]


@pytest.mark.parametrize("input_file, raw_data, sample", test_args)
def test_initial_and_v2_pipelines_match(input_file, raw_data, sample, v2_arg_parser):
    """Setting this up to be parametrized

    NOTE: It would be great if we could turn off printing on the initial pipeline version.
    Right now it will put outputs under the ./data/clean/RUN folder while testing, same as with the usual run

    See: https://docs.pytest.org/en/7.1.x/example/parametrize.html
    """
    data = load_to_pd(raw_data, input_file)

    # TODO: Ok we get a mismatch with 10,000 rows
    if sample:
        data = data.sample(n=10000, random_state=1)

    argv = ["--input_file", input_file, "--raw_data", raw_data]

    # I don't think args are used after initial setup
    # if they are needed, extract create parser to it's own function as in v2 pipeline and repeat below process
    initial_output, initial_misc, _ = initial_process_data(data)

    new_options = vars(v2_arg_parser.parse_args(argv))
    new_normalized, new_misc, _ = new_process_data(data, **new_options)

    new_misc = new_misc.sort_index().reset_index(drop=True)
    initial_misc = initial_misc.sort_index().reset_index(drop=True)

    mismatch_output_columns = set(new_misc.columns) ^ set(initial_misc.columns)
    assert len(mismatch_output_columns) == 0, "Output columns are mismatching"
    if not new_misc.equals(initial_misc):
        # TODO: write these files somewhere reasonable
        # Note: keep_equal is weird...you need to keep_shape and then filter by the index of the rows
        # with discrepancies in order to get the full rows
        differences_only = new_misc.compare(initial_misc)
        differences_only.to_csv("differences_only.csv")
        differences = new_misc.compare(
            initial_misc, keep_equal=True, keep_shape=True
        ).loc[differences_only.index.tolist()]
        initial_misc.to_csv("initial_output.csv")
        new_misc.to_csv("v2_output.csv")
        differences.to_csv("differences.csv")
        assert new_misc.equals(
            initial_misc
        ), "Output columns seem to match but other issue in dataframe"

    mismatch_misc_columns = set(new_normalized.columns) ^ set(initial_output.columns)
    assert len(mismatch_misc_columns) == 0, "Misc columns are mismatching"

    if not new_normalized.equals(initial_output):
        differences = new_normalized.compare(initial_output)
        print("Differences in misc DataFrames:")
        print(differences)
        assert new_normalized.equals(
            initial_output
        ), "Misc columns seem to match but other issue in dataframe"
