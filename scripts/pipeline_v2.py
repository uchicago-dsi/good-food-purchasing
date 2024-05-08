import argparse

import pandas as pd

from cgfp.util import load_to_pd, save_pd_to_csv


DEFAULT_INPUT_FILE = "CONFIDENTIAL_CGFP bulk data_073123.xlsx"
DEFAULT_MISC_FILE = "misc.csv"
CLEAN_FILE_PREFIX = "clean_"


def create_parser():
    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument(
        "--input_file", default=DEFAULT_INPUT_FILE, help="Input file path"
    )
    parser.add_argument("--clean_folder", default="./data/clean/", help="")
    parser.add_argument(
        "--clean_file",
        default=None,
        help="Clean file path. If not specified, it will be automatically generated based on the input file.",
    )
    parser.add_argument(
        "--misc_file", default=DEFAULT_MISC_FILE, help="Miscellaneous file path"
    )
    parser.add_argument(
        "--raw_data", default="./data/raw/", help="Relative path to raw data directory"
    )
    parser.add_argument(
        "--disable-output", action="store_false", dest="do_write_output", default=True
    )
    return parser


def process_data(df, **options):
    # super messy fill in the blanks on these dataframes

    # main df processing

    df_processed = df.copy()

    df_processed["Basic Type"] = df_processed["Product Name"]
    output_column_names = [
        "Product Type",
        "Food Product Group",
        "Food Product Category",
        "Primary Food Product Category",
        "Product Name",
        "Basic Type",
        "Sub-Type 1",
        "Sub-Type 2",
        "Flavor/Cut",
        "Shape",
        "Skin",
        "Seed/Bone",
        "Processing",
        "Cooked/Cleaned",
        "WG/WGR",
        "Dietary Concern",
        "Additives",
        "Dietary Accommodation",
        "Frozen",
        "Packaging",
        "Commodity",
    ]

    missing_columns = [
        column for column in output_column_names if column not in df_processed.columns
    ]
    for column in missing_columns:
        df_processed[column] = None

    extra_columns = [
        column for column in df_processed.columns if column not in output_column_names
    ]
    df_processed.drop(extra_columns, axis=1, inplace=True)

    # misc df processing

    misc_column_names = [
        "Product Type",
        "Food Product Group",
        "Food Product Category",
        "Basic Type",
        "Sub-Type 1",
        "Sub-Type 2",
        "Misc",
    ]
    misc = pd.DataFrame(columns=misc_column_names)

    return misc, df_processed[output_column_names]


def main(argv):
    # input
    parser = create_parser()
    options = vars(parser.parse_args(argv))

    # processing
    df_loaded = load_to_pd(**options)
    misc, df_processed = process_data(df_loaded, **options)

    # output
    save_pd_to_csv(df_processed, **options)
    save_pd_to_csv(misc, options.get("clean_folder"), options.get("misc_file"))


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
