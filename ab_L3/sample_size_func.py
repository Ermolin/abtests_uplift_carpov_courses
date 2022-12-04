import numpy as np
import pandas as pd
from scipy.stats import norm

def estimate_sample_size(df, metric_name, effects, alpha=0.05, beta=0.2):
    results = list()
    mean = np.mean(df[metric_name].values)
    std = np.std(df[metric_name].values)

    for epsilon in effects:
        abs_effect = mean - mean* epsilon
        disp_sum = 2 * (std ** 2)

        f_alpha = abs(norm.ppf( alpha/2 , loc=0, scale=1))
        f_beta =  norm.ppf(1- beta , loc=0, scale=1)

        n = ((f_alpha+f_beta)**2) * disp_sum / (abs_effect ** 2 )
        n = float(np.ceil(n))
        results.append((epsilon, n))
        
    return pd.DataFrame.from_records(results,columns=['effect', 'sample_size'])