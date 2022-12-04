import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


def estimate_second_type_error(df_pilot_group, df_control_group, metric_name, effects, alpha=0.05, n_iter=10000, seed=None):
    """Оцениваем ошибки второго рода.

    Бутстрепим выборки из пилотной и контрольной групп тех же размеров, добавляем эффект к пилотной группе,
    считаем долю случаев без значимых отличий.
    
    df_pilot_group - pd.DataFrame, датафрейм с данными пилотной группы
    df_control_group - pd.DataFrame, датафрейм с данными контрольной группы
    metric_name - str, названия столбца с метрикой
    effects - List[float], список размеров эффектов ([1.03] - увеличение на 3%).
    alpha - float, уровень значимости для статтеста
    n_iter - int, кол-во итераций бутстрапа
    seed - int or None, состояние генератора случайных чисел

    return - dict, {размер_эффекта: ошибка_второго_рода}
    """
    np.random.seed(seed)
    results = {}
    # bootstrap data samples
    pilot_df_bootstrap =  np.random.choice(df_pilot_group[metric_name],size=(len(df_pilot_group[metric_name]),n_iter))
    control_df_bootstrap = np.random.choice(df_control_group[metric_name],size=(len(df_control_group[metric_name]),n_iter))
    for effect in effects:
        # ttest them
        s, p= ttest_ind(a=pilot_df_bootstrap*effect,
              b=control_df_bootstrap,
              axis=0,
              equal_var=True,
              alternative='two-sided')
        results[effect] = p[p>alpha].shape[0]/s.shape[0]
    return results