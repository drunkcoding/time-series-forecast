import pandas as pd
from forecast.data.loader import DataParser

class ProphetDataParser(DataParser):
    def __init__(self) -> None:
        pass

    def format_sndlib_xml(self, path):
        df = self.parse_sndlib_xml(path)
        node_pairs = df.columns.values.tolist()
        # print(node_pairs)
        node_pairs.remove("timestamps")

        records = {}

        for pair in node_pairs:
            df_sub = df[["timestamps", pair]]
            df_sub = df_sub.rename(columns={"timestamps": "ds", pair: "y"})
            records[pair] = df_sub

        return records

class MultivariateDataParser(DataParser):
    def __init__(self) -> None:
        pass

    def format_sndlib_xml(self, path):
        df = self.parse_sndlib_xml(path)
        node_pairs = df.columns.values.tolist()
        # print(node_pairs)
        node_pairs.remove("timestamps")

        records = []

        for pair in node_pairs:
            df_sub = df[["timestamps", pair]]
            df_sub = df_sub.rename(columns={"timestamps": "ds", pair: "y"})
            records.append(df_sub)

        return pd.concat(records, ignore_index=True)

if __name__ == "__main__":
    parser = ProphetDataParser()
    df = parser.format_sndlib_xml("directed-abilene-zhang-5min-over-6months-ALL")
    print(df)