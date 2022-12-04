import pandas as pd
import numpy as np
import datetime
import hashlib

class ABSplitter:
    def __init__(self, count_slots, salt_one, salt_two):
        self.count_slots = count_slots
        self.salt_one = salt_one
        self.salt_two = salt_two

        self.slots = np.arange(count_slots)
        self.experiments = []
        self.experiment_to_slots = dict()
        self.slot_to_experiments = dict()

    def split_experiments(self, experiments):
        self.experiments = experiments
        experiments = sorted(
            experiments,
            key=lambda x: len(x['conflict_experiments']),
            reverse=True
        )

        experiments_without_place = []
        slot_to_experiments = {slot: [] for slot in self.slots}
        experiment_to_slots = {experiment['experiment_id']: [] for experiment in experiments}
        for experiment in experiments:
            # найдём доступные слоты
            notavailable_slots = []
            for conflict_experiment_id in experiment['conflict_experiments']:
                notavailable_slots += experiment_to_slots[conflict_experiment_id]
            available_slots = list(set(self.slots) - set(notavailable_slots))

            if experiment['count_slots'] > len(available_slots):
                experiments_without_place.append(experiment)
                continue

            # shuffle - чтобы внести случайность, иначе они все упорядочены будут по номеру slot
            np.random.shuffle(available_slots)
            available_slots_orderby_count_experiment = sorted(
                available_slots,
                key=lambda x: len(slot_to_experiments[x]), reverse=True
            )
            experiment_slots = available_slots_orderby_count_experiment[:experiment['count_slots']]
            experiment_to_slots[experiment['experiment_id']] = experiment_slots
            for slot in experiment_slots:
                slot_to_experiments[slot].append(experiment['experiment_id'])
        self.slot_to_experiments = slot_to_experiments
        self.experiment_to_slots = experiment_to_slots
        return experiments_without_place
        
    # add extra function for hash  (str+ salt) % modulo spliting
    def get_hash_modulo(self,
                         value: str,
                         modulo: int,
                         salt: int):
        
        hash_value = int(hashlib.md5(str.encode(value)).hexdigest(), 16)
        return (hash_value + salt) % modulo
    
    
    def process_user(self, user_id: str):
        slot = self.get_hash_modulo(user_id, self.count_slots, self.salt_one)
        experiments = self.slot_to_experiments[slot]
        
        # agg experiments from list of exp_ids we take from self.slot_to_experiments dict
        all_experimens_w_slot = [
            experiment for experiment in self.experiments
            if experiment['experiment_id'] in experiments
        ]
        
        twice_splited_groups=[]
        for experiment in all_experimens_w_slot:
            group = 'pilot' if self.get_hash_modulo(user_id + experiment['experiment_id'], 2, self.salt_two) else 'control'
            twice_splited_groups.append((experiment['experiment_id'], group))
        
        return (slot, twice_splited_groups)
    
        