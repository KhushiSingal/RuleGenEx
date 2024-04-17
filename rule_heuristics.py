import operator
import numpy as np

from functools import reduce
from .bitarray_backend import PythonIntArray, BitArray


class RuleHeuristics:
    def __init__(self, X, y, classes_, condition_map,
                 cov_threshold=0.0, conf_threshold=0.5,
                 bitarray_backend='python-int'):
        self.X = X
        self.y = y
        self.classes_ = classes_
        self.n_classes = len(classes_)
        self.condition_map = condition_map
        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold

        if bitarray_backend == 'python-int':
            self.bitarray_ = PythonIntArray(X.shape[0], classes_)
        else:
            self.bitarray_ = BitArray(X.shape[0], classes_)

        self.training_bit_sets = None
        self._cond_cov_dict = None

        self.training_heuristics_dict = None

        self.ones = self.bitarray_.generate_ones()
        self.zeros = self.bitarray_.generate_zeros()

    def get_conditions_heuristics(self, conditions,
                                  not_cov_mask=None):
        empty_list = [0.0] * self.n_classes
        heuristics_dict = {'cov_set': [self.zeros] * (self.n_classes + 1),
                           'cov': 0.0,
                           'cov_count': 0.0,
                           'class_cov_count': empty_list,
                           'conf': empty_list,
                           'supp': empty_list}
        if len(conditions) == 0:
            return self.get_training_heuristics_dict(
                not_cov_mask=not_cov_mask), not_cov_mask

        b_array_conds = [reduce(operator.and_,
                                [self._cond_cov_dict[i][cond[0]] for cond in
                                 conditions])
                         for i in range(self.n_classes)]

        b_array_conds.append(reduce(operator.or_, [i for i in b_array_conds]))

        cov_count = self.bitarray_.get_number_ones(b_array_conds[-1])

        if cov_count == 0:
            return heuristics_dict, not_cov_mask

        class_cov_count = [self.bitarray_.get_number_ones(b_array_conds[i]) for
                           i in
                           range(self.n_classes)]
        coverage = cov_count / self.X.shape[0]
        heuristics_dict['cov_set'] = b_array_conds
        heuristics_dict['cov'] = coverage
        heuristics_dict['cov_count'] = cov_count
        heuristics_dict['class_cov_count'] = class_cov_count
        heuristics_dict['conf'] = [class_count / cov_count for class_count in
                                   class_cov_count]
        heuristics_dict['supp'] = [class_count / self.X.shape[0] for class_count
                                   in class_cov_count]

        return heuristics_dict, not_cov_mask

    def compute_rule_heuristics(self, ruleset, not_cov_mask=None,
                                sequential_covering=False, recompute=False):
        if recompute:
            for rule in ruleset:
                heuristics_dict, _ = self.get_conditions_heuristics(
                    rule.A)
                rule.set_heuristics(heuristics_dict)
            return

        if not_cov_mask is None:
            not_cov_mask = self.ones

        if sequential_covering:
            accurate_rules = []
            local_not_cov_samples = not_cov_mask
            for rule in ruleset:

                result, not_cov_samples_with_rule = self.rule_is_accurate(
                    rule,
                    local_not_cov_samples)
                if result:
                    accurate_rules.append(rule)
                    local_not_cov_samples = not_cov_samples_with_rule

            ruleset.rules[:] = accurate_rules

        else:
            for rule in ruleset:
                self.set_rule_heuristics(rule, not_cov_mask)

    def _compute_training_bit_sets(self):
        training_bit_set = [
            self.bitarray_.get_array(self.y == self.classes_[i]) for
            i in range(self.n_classes)]
        training_bit_set.append(reduce(operator.or_,
                                       training_bit_set))

        return training_bit_set

    def _compute_condition_bit_sets(self):
        cond_cov_dict = [{} for _ in range(self.n_classes + 1)]
        for cond_id, cond in self.condition_map.items():
            cond_coverage_bitarray = self.bitarray_.get_array(
                cond.satisfies_array(self.X))
            
            for i in range(self.n_classes):
                cond_cov_dict[i][cond_id] = cond_coverage_bitarray & \
                                            self.training_bit_sets[i]
            cond_cov_dict[-1][cond_id] = cond_coverage_bitarray
        return cond_cov_dict

    def initialize_sets(self):
        self.training_bit_sets = self._compute_training_bit_sets()
        self._cond_cov_dict = self._compute_condition_bit_sets()

    def rule_is_accurate(self, rule, not_cov_samples):
        if self.bitarray_.get_number_ones(not_cov_samples) == 0:
            return False, not_cov_samples

        local_not_cov_samples = self.set_rule_heuristics(rule,
                                                             not_cov_samples)

        if rule.conf > self.conf_threshold and rule.cov > self.cov_threshold:
            return True, local_not_cov_samples
        else:
            return False, not_cov_samples

    def create_empty_heuristics_dict(self):
        empty_list = [0.0] * self.n_classes
        return {'cov_set': [self.zeros] * (self.n_classes + 1),
                'cov': 0.0,
                'cov_count': 0.0,
                'class_cov_count': empty_list,
                'conf': empty_list,
                'supp': empty_list}

    def get_training_heuristics_dict(self, not_cov_mask=None):
        if self.training_heuristics_dict is None:
            cov_count = self.bitarray_.get_number_ones(
                self.training_bit_sets[-1])
            class_cov_count = [
                self.bitarray_.get_number_ones(self.training_bit_sets[i]) for i
                in
                range(self.n_classes)]
            coverage = cov_count / self.X.shape[0]
            train_heur_dict = {'cov_set': self.training_bit_sets,
                               'cov': coverage,
                               'cov_count': cov_count,
                               'class_cov_count': class_cov_count,
                               'conf': [class_count / cov_count for class_count
                                        in
                                        class_cov_count],
                               'supp': [class_count / self.X.shape[0] for
                                        class_count
                                        in class_cov_count]}
            self.training_heuristics_dict = train_heur_dict
        if not_cov_mask is None:
            return self.training_heuristics_dict
        else:
            if self.bitarray_.get_number_ones(not_cov_mask) == 0:
                empty_list = [0.0] * self.n_classes
                return {'cov_set': [self.zeros] * (self.n_classes + 1),
                        'cov': 0.0,
                        'cov_count': 0.0,
                        'class_cov_count': empty_list,
                        'conf': empty_list,
                        'supp': empty_list}
            masked_training_heuristics = [b_array_measure & not_cov_mask for
                                          b_array_measure in
                                          self.training_bit_sets]
            cov_count = self.bitarray_.get_number_ones(
                masked_training_heuristics[-1])
            class_cov_count = [
                self.bitarray_.get_number_ones(masked_training_heuristics[i])
                for i in
                range(self.n_classes)]
            coverage = cov_count / self.X.shape[0]

            return {'cov_set': masked_training_heuristics,
                    'cov': coverage,
                    'cov_count': cov_count,
                    'class_cov_count': class_cov_count,
                    'conf': [class_count / cov_count for class_count
                             in
                             class_cov_count],
                    'supp': [class_count / self.X.shape[0] for
                             class_count
                             in class_cov_count]}

    def combine_heuristics(self, heuristics1, heuristics2):
        cov_set = [self.zeros] * (self.n_classes + 1)
        for i in range(self.n_classes + 1):
            cov_set[i] = heuristics1['cov_set'][i] & heuristics2['cov_set'][i]
        cov_count = self.bitarray_.get_number_ones(cov_set[-1])

        class_cov_count = [self.bitarray_.get_number_ones(cov_set[i]) for i in
                           range(self.n_classes)]

        coverage = cov_count / self.X.shape[0]
        if coverage == 0:
            empty_list = [0.0] * self.n_classes
            return {'cov_set': cov_set,
                    'cov': 0.0,
                    'cov_count': 0.0,
                    'class_cov_count': class_cov_count,
                    'conf': empty_list,
                    'supp': empty_list}
        return {'cov_set': cov_set,
                'cov': coverage,
                'cov_count': cov_count,
                'class_cov_count': class_cov_count,
                'conf': [class_count / cov_count for class_count
                         in
                         class_cov_count],
                'supp': [class_count / self.X.shape[0] for
                         class_count
                         in class_cov_count]

                }

    def set_rule_heuristics(self, rule, mask):
        mask_cov_set = [cov_set & mask
                        for cov_set in rule.heuristics_dict['cov_set']]

        cov_count = self.bitarray_.get_number_ones(mask_cov_set[-1])

        if cov_count == 0:
            rule.conf = 0.0
            rule.supp = 0.0
            rule.cov = 0.0
            return self.bitarray_.get_complement(mask_cov_set[-1],
                                                 self.ones) & mask

        else:
            class_cov_count = [self.bitarray_.get_number_ones(mask_cov_set[i])
                               for i in
                               range(self.n_classes)]

            coverage = cov_count / self.X.shape[0]

            rule.conf = class_cov_count[rule.class_index] / cov_count
            rule.supp = class_cov_count[rule.class_index] / self.X.shape[0]
            rule.cov = coverage
            rule.n_samples = np.array(class_cov_count)

            return self.bitarray_.get_complement(mask_cov_set[-1],
                                                 self.ones) & mask

