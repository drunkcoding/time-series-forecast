import glob
import re
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime


class DataParser:
    def __init__(self) -> None:
        pass

    def parse_sndlib_xml(self, path):

        data_folder = os.path.join(path, "*.xml")

        # Get node list
        pattern = r'<node id="([A-Za-z0-9\.]+)">'
        example_file = glob.glob(data_folder)[0]
        with open(example_file, "r") as f:
            data = f.read()
            all_groups = re.findall(pattern, data)
            nodes = all_groups

        records = {f"{i}_{j}": list() for i in nodes for j in nodes if i != j}
        timestamps = []

        pattern = r'<demand id="([_A-Za-z0-9\.]+)">\n   <source>([A-Za-z0-9\.]+)</source>\n   <target>([A-Za-z0-9\.]+)</target>\n   <demandValue> (\d+.\d+) </demandValue>'

        for name in tqdm(glob.glob(data_folder)):
            timestamp = name.split(".")[0]
            timestamp = timestamp.split("-")[-2:]
            timestamp = " ".join(timestamp)

            datetime_obj = datetime.strptime(timestamp, "%Y%m%d %H%M")
            # datetime_str = datetime.strftime(datetime_obj, "%Y-%m-%d %H:%M:%S")

            timestamps.append(datetime_obj)

            with open(name, "r",) as f:
                data = f.read()

            for id in records:
                records[id].append(None)

            all_groups = re.findall(pattern, data)
            for group in all_groups:
                id, source, target, value = group
                records[id][-1] = float(value)

        df = pd.DataFrame(records)
        # node_pairs = df.columns
        df['timestamps'] = timestamps

        return df


if __name__ == "__main__":
    parser = DataParser()
    df = parser.parse_sndlib_xml("directed-abilene-zhang-5min-over-6months-ALL")
    print(df)