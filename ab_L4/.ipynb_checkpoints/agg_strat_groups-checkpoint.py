import numpy as np
import pandas as pd
import random

# generate weigths from slice if they are not passed 
def generate_weigths(data,strat_columns):
    #data_not_strat_cols = [i for i in data.columns if i not in strat_columns]
    lenght = data.shape[0]
    dt = data.groupby(strat_columns).agg(count=(strat_columns[0], 'count')).transpose().to_dict()
    for key, val in dt.items():
        dt[key]=dt[key]['count'] / lenght
    return dt

# get indexes by strat's
def get_indexes_dict(weights, data, strat_columns):
    if len(strat_columns)>1 :
        indexes_dict={}
        for key, value in weights.items():
            indexes = data.index
            for val, col in zip(key, strat_columns):
                indexes = data.loc[indexes][data[col] == val].index
            indexes_dict[key]= indexes
    else:
        indexes = data.index
        indexes_dict={}
        for val in data[strat_columns[0]].unique():
            indexes = data.loc[indexes][data[strat_columns] == val].index
            indexes_dict[strat_columns[0]]= indexes
    return indexes_dict

# agg group with stratification
def get_group(indexes_dict, data, group_size, weights):
    if indexes_dict:
        inds = []
        for strat , indexes in indexes_dict.items():
            inds+=list(data.loc[indexes].sample(n = int(np.ceil(group_size * weights[strat]))).index)
        inds = random.sample(inds, group_size)
    else:
        inds = np.random.choice(list(data.index), group_size, replace=False)
    return data.loc[inds]

# final function
def select_stratified_groups(data, strat_columns, group_size, weights=None, seed=None):
    if seed:
        np.random.seed(seed = seed)
    if not weights:
        weights = generate_weigths(data,strat_columns)
    
    if len(strat_columns)>1 :
        indexes_dict = get_indexes_dict(weights, data, strat_columns)
        data_pilot = get_group(indexes_dict, data, group_size, weights)
        data_control = get_group(indexes_dict, data, group_size, weights)
    else:
        data_pilot = get_group(None, data, group_size, weights)
        data_control = get_group(None, data, group_size, weights)
    return (data_pilot, data_control)