import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def estimate_first_type_error(df_pilot_group, df_control_group, metric_name, alpha=0.05, n_iter=10000, seed=None):
    """Оцениваем ошибку первого рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, считаем долю случаев с значимыми отличиями.
    
    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел.

    return - float, ошибка первого рода
    """
    
    np.random.seed(seed)
    # bootstrap data samples
    pilot_df_bootstrap =  np.random.choice(df_pilot_group[metric_name],size=(len(df_pilot_group[metric_name]),n_iter))
    control_df_bootstrap = np.random.choice(df_control_group[metric_name],size=(len(df_control_group[metric_name]),n_iter))
    # ttest them
    s, p= ttest_ind(a=pilot_df_bootstrap,
          b=control_df_bootstrap,
          axis=0,
          equal_var=True,
          alternative='two-sided')
    return p[p<alpha].shape[0]/s.shape[0]