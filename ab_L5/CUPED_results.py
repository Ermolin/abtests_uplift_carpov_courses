import numpy as np
import pandas as pd
import datetime

def calculate_metric(
    df, value_name, user_id_name, list_user_id, date_name, period, metric_name
):
    # date mask
    df[date_name] = pd.to_datetime(df[date_name], format="%Y-%m-%d")
    df = df[(df[date_name] >= datetime.datetime.strptime(period['begin'], "%Y-%m-%d"))&
            (df[date_name] < datetime.datetime.strptime(period['end'], "%Y-%m-%d"))]
    
    # users mask
    df = df[df[user_id_name].isin(list_user_id)]
    
    # agg metric
    #agg_df = df.groupby(user_id_name).agg(
    #metric_name = pd.NamedAgg(column=value_name, aggfunc=metric_name))
    agg_df = df.groupby(user_id_name)[[value_name]].sum()\
    .rename(columns={value_name: metric_name})\
    .reset_index()
    
    full_agg_df = pd.merge(pd.DataFrame(list_user_id, columns=[user_id_name]), agg_df, on=user_id_name, how='outer')
    full_agg_df = full_agg_df.fillna(0)
    return full_agg_df

def calculate_metric_cuped(
    df, value_name, user_id_name, list_user_id, date_name, periods, metric_name
):
    df_prepilot = calculate_metric(
        df, value_name, user_id_name, list_user_id, date_name,
        periods['prepilot'], metric_name
    ).rename(columns={metric_name: f'{metric_name}_prepilot'})
    df_pilot = calculate_metric(
        df, value_name, user_id_name, list_user_id, date_name,
        periods['pilot'], metric_name
    )
    df = pd.merge(
        df_prepilot,
        df_pilot,
        on=user_id_name
    )
    # CUPED
    target_values = df[metric_name].values
    covariate_values = df[f'{metric_name}_prepilot'].values
    covariance = np.cov(target_values, covariate_values)[0, 1]
    variance = covariate_values.var()
    theta = covariance / variance
    df[f'{metric_name}_cuped'] = (
        target_values - theta * covariate_values
    )
    return df