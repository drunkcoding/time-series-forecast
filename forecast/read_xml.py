from distutils.filelist import findall
import glob
import re
import pandas as pd
from tqdm import tqdm
import missingno as msno
import matplotlib.pyplot as plt
from datetime import datetime


pattern = r'<node id="([A-Za-z0-9]+)">'
with open("directed-abilene-zhang-5min-over-6months-ALL/demandMatrix-abilene-zhang-5min-20040301-0000.xml", "r") as f:
    data = f.read()
    all_groups = re.findall(pattern, data)
    nodes = all_groups

pairs = [
    f"{i}_{j}"
    for i in nodes
    for j in nodes
    if i != j
]

records = {
    f"{i}_{j}": list()
    for i in nodes
    for j in nodes
    if i != j
}

pattern = r'<demand id="([_A-Za-z0-9]+)">\n   <source>([A-Za-z0-9]+)</source>\n   <target>([A-Za-z0-9]+)</target>\n   <demandValue> (\d+.\d+) </demandValue>'

for name in tqdm(glob.glob("directed-abilene-zhang-5min-over-6months-ALL/*.xml")):
    # print(name)

    timestamp = name.split(".")[0]
    timestamp = timestamp.split("-")[-2:]
    timestamp = " ".join(timestamp)

    datetime_object = datetime.strptime(timestamp, '%Y%m%d %H%M')

    # Reading the data inside the xml
    # file to a variable under the name
    # data
    with open(name, "r",) as f:
        data = f.read()

    for id in records:
        records[id].append(None)

    all_groups = re.findall(pattern, data)
    for group in all_groups:
        id, source, target, value = group
        # if not id in records:
        #     records[id] = []
        records[id][-1] = float(value)

for id in records:
    print(id, len(records[id]))

df = pd.DataFrame(records)

msno.heatmap(df, labels=False)
plt.savefig("msno.png")

df['time'] = [x for x in range(len(df.index))]
df = pd.melt(df, id_vars=['time'], value_vars=pairs)
df.to_csv("abilene.csv", index=False)
