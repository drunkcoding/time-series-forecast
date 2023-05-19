import numpy as np
import pandas as pd
import scipy.stats as stats
from dataclasses import dataclass, field
import os
from transformers import HfArgumentParser
import statsmodels.api as sm

@dataclass
class ModelConfig:
    data: str = field(metadata={"help": "path to csv data"})

def z_score_norm(x):
    return (x - np.mean(x)) / np.std(x)

def z_score_reverse(x, mean, std):
    return x * std + mean

# ks test where the null hypothesis is that the data is normally distributed with mean 0 and std 1
def ks_norm_test(x):
    return stats.kstest(x, 'norm', args=(0, 1))

def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

if __name__ == "__main__":
    parser = HfArgumentParser(ModelConfig)
    args = parser.parse_args_into_dataclasses()[0]

    # load data
    df = pd.read_csv(args.data)
    df = df.drop(columns=["time"])

    columns = df.columns.values
    print(columns)

    sample_od_data = df[columns[1]].to_numpy()
    print(stats.describe(sample_od_data))

    sample_od_data = sample_od_data[sample_od_data > 0]
    for i in range(50, 101):
        clip_value = np.percentile(sample_od_data, i)
        clipped_data = sample_od_data[sample_od_data < clip_value]
        outliers = sample_od_data[sample_od_data >= clip_value]
        clipped_data_norm = z_score_norm(clipped_data)
        outliers_norm = z_score_norm(outliers)
        res_clip = ks_norm_test(clipped_data_norm)  
        res_outlier = ks_norm_test(outliers_norm)
        print("clip", i, res_clip.pvalue, res_clip.statistic, res_outlier.pvalue, res_outlier.statistic)

        # acf auto correlation function
        max_corr = 0
        max_lag = 0
        for lag in range(30,300):
            acf_clipped_data = sm.tsa.acf(clipped_data_norm, nlags=lag)
            acf_outliers = sm.tsa.acf(outliers_norm, nlags=lag)
            if acf_clipped_data[-1] > max_corr:
                max_corr = acf_clipped_data[-1]
                max_lag = lag
        print("autocorr", max_lag, max_corr)
    # # drop columns that are all 0
    # df = df.loc[:, (df != 0).any(axis=0)]

def min_max_norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def min_max_reverse(x, min, max):
    return x * (max - min) + min

def log_norm(x):
    return np.log(x)

def log_reverse(x):
    return np.exp(x)

def box_cox_norm(x):
    return stats.boxcox(x)[0]

def box_cox_reverse(x, lambda_):
    return stats.boxcox(x, lambda_)[0]