import sys

import polars as pl


def read_excel_write_csv(in_, out):
    df = pl.read_excel(in_, read_csv_options={"infer_schema_length": 1, "null_values": ["NULL"]})
    df.write_csv(out)


if __name__ == "__main__":
    read_excel_write_csv(sys.argv[1], sys.argv[2])
