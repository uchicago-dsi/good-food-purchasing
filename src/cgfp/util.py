"""Utils for CGFP pipline"""

from pathlib import Path

import pandas as pd

from cgfp.constants.pipeline_constants import CLEAN_FILE_PREFIX


def load_to_pd(raw_data: str, input_file: str, **options) -> pd.DataFrame:
    """Loads the data with minimal transformations and returns a DataFrame.

    Assumes input data fits comfortably in memory.

    Args:
        raw_data: The directory containing the raw data.
        input_file: The name of the input file.
        **options: Additional keyword arguments for future extensions.

    Returns:
        The loaded DataFrame.
    """
    INPUT_PATH = Path(raw_data) / input_file
    if not INPUT_PATH.exists():
        raise GoodFoodDataException(f"Could not find input file {INPUT_PATH}")
    df_raw = (
        pd.read_excel(INPUT_PATH, sheet_name=options.get("sheet_name", 0), dtype=str)
        if INPUT_PATH.suffix in [".xls", ".xlsx"]
        else pd.read_csv(INPUT_PATH, dtype=str)
    )
    # Note: Sometimes this is inconsistent — make sure it has "Product Type" column
    df_raw["Product Type"] = (
        df_raw["Normalized Product Type"] if "Normalized Product Type" in df_raw.columns else df_raw["Product Type"]
    )
    return df_raw


def save_pd_to_csv(
    df,
    clean_folder,
    do_write_output,
    output_file=None,
    input_file=None,
    **options,
) -> None:
    """Writes the DataFrame to a CSV file with the specified or automatically generated name.

    Args:
        df: The DataFrame to save.
        clean_folder: The directory to save the cleaned data.
        do_write_output: Flag to determine whether to write the output.
        output_file: The name of the output file. If not provided, it will be generated based on the input file.
        input_file: The name of the input file used to generate the output file name if output_file is not provided.
        **options: Additional keyword arguments for future extensions.

    Returns:
        None
    """
    if not output_file and not input_file:
        raise GoodFoodDataException(
            f"Not enough information to write data to {clean_folder}. Provide either output_file or input_file to be modified."
        )
    if not output_file:
        output_file = CLEAN_FILE_PREFIX + input_file

    output_path = Path(output_file)
    filename = output_path.stem
    ext = output_path.suffix

    if ext.lower() != ".csv":
        output_file = filename + ".csv"

    run_folder_path = options.get("run_folder_path")

    clean_file_path = run_folder_path / output_file
    clean_file_path = Path(str(clean_file_path).replace(" ", "_"))
    if do_write_output:
        df.to_csv(clean_file_path, index=False)


class GoodFoodBaseException(RuntimeError):
    """All returned exceptions should extend this"""

    pass


class GoodFoodDataException(GoodFoodBaseException):
    """Catchall for data quality and filesystem errors"""

    pass
