import numpy as np
import pandas as pd
import datetime

def filter_df( df, value_name, user_id_name, list_user_id, date_name, period, metric_name):
    df_filtered = (
        df[df[user_id_name].isin(list_user_id)
            & (df[date_name] >= period['begin'])
            & (df[date_name] < period['end'])])
    return df_filtered



def calculate_linearized_metric(
    df, value_name, user_id_name, list_user_id, date_name, period, metric_name, kappa=None
):
    df_filtered = filter_df( df, value_name, user_id_name, list_user_id, date_name, period, metric_name)
    
    df_full = pd.DataFrame({user_id_name: list_user_id})
    
    df_agg = (
        df_filtered
        .groupby(user_id_name)[[value_name]].agg(['sum', 'count'])
    )
    df_agg.columns = ['sum', 'count']
    
    if kappa is None:
        kappa = df_agg['sum'].sum() / df_agg['count'].sum()
    df_agg[metric_name] = df_agg['sum'] - kappa * df_agg['count']
    
    result = pd.merge(
        df_full,
        df_agg[[metric_name]].reset_index(),
        on=user_id_name,
        how='outer'
    ).fillna(0)
    return result