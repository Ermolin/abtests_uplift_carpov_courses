import numpy as np

def get_bernoulli_confidence_interval(values: np.array):
    """Вычисляет доверительный интервал для параметра распределения Бернулли.
    :param values: массив элементов из нулей и единиц.
    :return (left_bound, right_bound): границы доверительного интервала.
    При решении используйте приближение z(α/2) =1.96.
    """
    # consts
    z_a_2 = 1.96
    # income vals
    n = values.shape[0]
    n1 = np.count_nonzero(values)
    n0 = n-n1
    # statistics
    x_mean = n1/n
    std = np.sqrt((n0/n)*np.square(0-x_mean)+ (n1/n)*np.square(1-x_mean))
    # results
    lb = x_mean - 1.96*(std/np.sqrt(n))
    rb = x_mean + 1.96*(std/np.sqrt(n))
    return (np.max(np.array([0,lb])), np.min(np.array([1,rb])))