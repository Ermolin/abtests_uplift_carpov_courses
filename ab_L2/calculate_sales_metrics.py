import numpy as np
import pandas as pd

def calculate_sales_metrics(df, cost_name, date_name, sale_id_name, period, filters=None):
    df[date_name] = pd.to_datetime(df[date_name])

    filter_ = ((df[date_name] >= pd.to_datetime(period['begin'])) 
               & (df[date_name] < pd.to_datetime(period['end'])))
    if filters is not None:
        for c, v in filters.items():
            filter_ &= df[c].isin(v)
    
    flt_df = df.loc[filter_, :]

    dates = pd.date_range(start=period['begin'], end=period['end'], freq='D')[:-1]
    results_df = pd.DataFrame(index=dates)

    agg_df = (flt_df
              .groupby(date_name)
              .agg(revenue=(cost_name, 'sum'),
                   number_purchases=(sale_id_name, 'nunique'), ))
    sales_agg_df = (flt_df
                    .groupby([date_name, sale_id_name], as_index=False)
                    .agg(sum_check=(cost_name, 'sum'),
                         number_items=(cost_name, 'count'), )
                    .groupby(date_name)
                    .agg(average_check=('sum_check', 'mean'),
                         average_number_items=('number_items', 'mean'), ))

    agg_df = agg_df.join(sales_agg_df, how='outer').astype(float)
    return results_df.join(agg_df, how='outer').fillna(0.0)