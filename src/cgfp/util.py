from pathlib import Path
import os

import pandas as pd

from cgfp.constants.pipeline_constants import RUN_FOLDER, CLEAN_FILE_PREFIX


# Files


def load_to_pd(raw_data: str, input_file: str, **options):
    """Loads the data with minimal transformations returning dataframe

    Assumes input data fits confortably in memory.
    """
    INPUT_PATH = Path(raw_data) / input_file
    if not INPUT_PATH.exists():
        raise GoodFoodDataException(f"Could not find input file {INPUT_PATH}")
    df = (
        pd.read_excel(INPUT_PATH)
        if INPUT_PATH.suffix in [".xls", ".xlsx"]
        else pd.read_csv(INPUT_PATH)
    )
    return df


def save_pd_to_csv(
    df: pd.DataFrame,
    clean_folder,
    do_write_output,
    output_file=None,
    input_file=None,
    **options,
):
    """Writes dataframe to csv with given name or automatically based on input"""
    if not output_file and not input_file:
        raise GoodFoodDataException(
            f"Not enough information to write data to {clean_folder}. Provide either output_file or input_file to be modified."
        )
    if not output_file:
        output_file = CLEAN_FILE_PREFIX + input_file

    filename, ext = os.path.splitext(output_file)
    if ext.lower() != ".csv":
        output_file = filename + ".csv"

    run_folder_path = Path(clean_folder) / RUN_FOLDER
    run_folder_path.mkdir(parents=True, exist_ok=True)

    clean_file_path = run_folder_path / output_file
    if do_write_output:
        df.to_csv(clean_file_path, index=False)


# Exceptions


class GoodFoodBaseException(RuntimeError):
    """All returned exceptions should extend this"""

    pass


class GoodFoodDataException(GoodFoodBaseException):
    """Catchall for data quality and filesystem errors"""

    pass
