import re

import numpy as np

with open('evaluation/arima.log', 'r') as f:
    lines = f.readlines()

pattern = r'Test MAE: (\d+\.\d+)'

mae_list = []
for line in lines:
    if 'Test MAE' in line:
        mae = float(re.findall(pattern, line)[0])
        mae_list.append(mae)
        print('ARIMA MAE: %.5f' % mae)

print('ARIMA MAE: %.5f' % np.sum(mae_list))