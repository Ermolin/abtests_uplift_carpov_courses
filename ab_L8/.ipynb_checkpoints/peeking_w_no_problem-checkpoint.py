import numpy as np
import pandas as pd
import scipy.stats as stats

class SequentialTester:
    def __init__(
        self, metric_name, time_column_name,
        alpha, beta, pdf_one, pdf_two
    ):
        
        self.metric_name = metric_name
        self.time_column_name = time_column_name
        self.alpha = alpha
        self.beta = beta
        self.pdf_one = pdf_one
        self.pdf_two = pdf_two

        self.lower_bound = np.log(beta / (1 - alpha))
        self.upper_bound = np.log((1 - beta) / alpha)

        self.sum_log_ll = 0 # initial z_cumsum we will renew after tests
        self.count_data = 0 # n data in tests (n of days)
        self.stop_test = False # shoud we stop test (stat significant effect found)
        self.results = None # result of test ( what hypothesis is accepted) 

    def preprocess_data(self, *list_data):
        # sort data and take target values for data in list_data
        list_values = [
            data.sort_values(self.time_column_name)[self.metric_name].values
            for data in list_data
        ]
        return list_values

    
    
    def _check_test(self, data_control, data_pilot):
        # get new functions and borders and check if test are stat significant
        # get data
        values_control, values_pilot = self.preprocess_data(data_control, data_pilot)
        len_new_data = len(values_control)
        delta_values = values_pilot - values_control
        
        # get pdf's values and get new metric  
        pdf_one_values = self.pdf_one(delta_values)
        pdf_two_values = self.pdf_two(delta_values)
        z_cumsum = np.cumsum(np.log(pdf_two_values / pdf_one_values)) + self.sum_log_ll
    
        # arrays with test terults
        indexes_lower = np.arange(len_new_data)[z_cumsum < self.lower_bound]
        indexes_upper = np.arange(len_new_data)[z_cumsum > self.upper_bound]
        first_index_lower = indexes_lower[0] if len(indexes_lower) > 0 else len_new_data
        first_index_upper = indexes_upper[0] if len(indexes_upper) > 0 else len_new_data
        
        # return first index where test show stat result or no result
        # + update len data, sum_log_ll
        if first_index_lower < first_index_upper:
            self.results = (0., self.count_data + first_index_lower + 1)
            self.stop_test = True
            return self.results
        elif first_index_lower > first_index_upper:
            self.results = (1., self.count_data + first_index_upper + 1)
            self.stop_test = True
            return self.results
        else:
            self.count_data += len_new_data
            self.sum_log_ll = z_cumsum[-1]
            self.results = (0.5, self.count_data)
            return self.results
    
    def run_test(self, data_control, data_pilot):
        # run test from 0 point
        return self._check_test(data_control, data_pilot)

    def add_data(self, data_control, data_pilot):
        # if we didnt stoped test add new data and calculate statistics
        if self.stop_test:
            return self.results
        return self._check_test(data_control, data_pilot)