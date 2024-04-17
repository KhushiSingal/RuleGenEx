import copy
import time
from abc import abstractmethod, ABCMeta
from ast import literal_eval
from math import sqrt
from operator import attrgetter

import numpy as np
import pandas as pd

import scipy.stats as st

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

from rulegenex.rules import Rule, RuleSet
from .rule_extraction import RuleExtractorFactory
from .rule_heuristics import RuleHeuristics


def _ensemble_type(ensemble):
    if isinstance(ensemble, (BaggingClassifier, RandomForestClassifier)):
        return 'bagging'
    elif isinstance(ensemble, GradientBoostingClassifier):
        return 'gbt'
    elif str(ensemble.__class__) == "<class 'xgboost.sklearn.XGBClassifier'>":
        try:
            from xgboost import XGBClassifier
        except ModuleNotFoundError:
            raise ModuleNotFoundError('If you want to use '
                                      'xgboost.sklearn.XGBClassifier '
                                      'ensembles you should install xgboost '
                                      'library.')
        return 'gbt'
    elif str(ensemble.__class__) == "<class 'lightgbm.sklearn.LGBMClassifier'>":
        try:
            from lightgbm import LGBMClassifier
        except ModuleNotFoundError:
            raise ModuleNotFoundError('If you want to use '
                                      'lightgbm.sklearn.LGBMClassifier '
                                      'ensembles you should install lightgbm '
                                      'library.')
        return 'gbt'
    elif str(
            ensemble.__class__) == "<class 'catboost.core.CatBoostClassifier'>":
        try:
            from catboost import CatBoostClassifier
        except ModuleNotFoundError:
            raise ModuleNotFoundError('If you want to use '
                                      'catboost.core.CatBoostClassifier '
                                      'ensembles you should install catboost '
                                      'library.')
        return 'gbt'
    else:
        raise NotImplementedError


def _statistical_error_estimate(N, e, z_alpha_half):
    numerator = e + (z_alpha_half ** 2 / (2 * N)) + z_alpha_half * sqrt(
        ((e * (1 - e)) / N) + (z_alpha_half ** 2 / (4 * N ** 2)))
    denominator = 1 + ((z_alpha_half ** 2) / N)
    return numerator / denominator


class BaseRuleGenEx(BaseEstimator, metaclass=ABCMeta):
    def __init__(self,
                 base_ensemble=None,
                 n_estimators=5,
                 tree_max_depth=3,
                 percent_training=None,
                 early_stop=0,
                 metric='f1',
                 float_threshold=-1e-6,
                 column_names=None,
                 random_state=None,
                 verbose=0):
        self.base_ensemble = base_ensemble
        self.n_estimators = n_estimators
        self.tree_max_depth = tree_max_depth
        self.percent_training = percent_training
        self.early_stop = early_stop
        self.metric = metric
        self.float_threshold = float_threshold
        self.column_names = column_names
        self.random_state = random_state
        self.verbose = verbose

    def _more_tags(self):
        return {'binary_only': True}

    def fit(self, X, y):
        pass

    @abstractmethod
    def _validate_and_create_base_ensemble(self):
        pass

    @abstractmethod
    def _remove_opposite_conditions(self, conditions, class_index):
        pass

    @abstractmethod
    def _initialize_sets(self):
        pass

    @abstractmethod
    def _add_default_rule(self, ruleset):
        pass

    @abstractmethod
    def _combine_rulesets(self, ruleset1, ruleset2):
        pass

    @abstractmethod
    def _sequential_covering_pruning(self, ruleset):
        pass

    @abstractmethod
    def _simplify_rulesets(self, ruleset):
        pass

    @abstractmethod
    def _evaluate_combinations(self, simplified_ruleset, combined_rules):
        pass


class RuleGenExClassifier(ClassifierMixin, BaseRuleGenEx):
    def __init__(self,
                 base_ensemble=None,
                 n_estimators=5,
                 tree_max_depth=3,
                 cov_threshold=0.0,
                 conf_threshold=0.5,
                 c=0.25,
                 percent_training=None,
                 early_stop=0,
                 metric='f1',
                 rule_order='supp',
                 sort_by_class=None,
                 float_threshold=1e-6,
                 column_names=None,
                 random_state=None,
                 verbose=0
                 ):
        super().__init__(base_ensemble=base_ensemble,
                         n_estimators=n_estimators,
                         tree_max_depth=tree_max_depth,
                         percent_training=percent_training,
                         early_stop=early_stop,
                         metric=metric,
                         float_threshold=float_threshold,
                         column_names=column_names,
                         random_state=random_state,
                         verbose=verbose
                         )

        self.cov_threshold = cov_threshold
        self.conf_threshold = conf_threshold
        self.c = c
        self.rule_order = rule_order
        self.sort_by_class = sort_by_class

    def fit(self, X, y):
        self._rule_extractor = None
        self._rule_heuristics = None
        self._base_ens_type = None
        self._weights = None
        self._global_condition_map = None
        self._bad_combinations = None
        self._good_combinations = None
        self._early_stop_cnt = 0
        self.alpha_half_ = None

        self.X_ = None
        self.y_ = None
        self.classes_ = None
        self.original_rulesets_ = None
        self.simplified_ruleset_ = None
        self.combination_time_ = None
        self.n_combinations_ = None
        self.ensemble_training_time_ = None

        # Check that X and y have correct shape
        if self.column_names is None:
            if isinstance(X, pd.DataFrame):
                self.column_names = X.columns
        X, y = check_X_y(X, y)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.alpha_half_ = st.norm.ppf(1 - (self.c / 2))

        if self.percent_training is None:
            self.X_ = X
            self.y_ = y
        else:
            x, _, y, _ = train_test_split(X, y,
                                          test_size=(1 - self.percent_training),
                                          shuffle=True, stratify=y,
                                          random_state=self.random_state)
            self.X_ = x
            self.y_ = y

        # add rule ordering funcionality (2023.03.30)
        self._sorting_list = ['cov', 'conf', 'supp']
        self._sorting_list.remove(self.rule_order)
        self._sorting_list.insert(0, self.rule_order)
        self._sorting_list.reverse()
        if self.sort_by_class is not None:
            if isinstance(self.sort_by_class, bool):
                self.sort_by_class = self.classes_.tolist()

        if self.n_estimators is None or self.n_estimators < 2:
            raise ValueError(
                "Parameter n_estimators should be at "
                "least 2 for using the RuleGenEx method.")

        if self.verbose > 0:
            print('Validating original ensemble...')
        try:
            if self.base_ensemble is None:
                self.base_ensemble_ = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            else:
                self.base_ensemble_ = self.base_ensemble
            self._base_ens_type = _ensemble_type(self.base_ensemble_)
        except NotImplementedError:
            print(
                f'Base ensemble of type {type(self.base_ensemble_).__name__} '
                f'is not supported.')
        try:
            check_is_fitted(self.base_ensemble_)
            self.ensemble_training_time_ = 0
            if self.verbose > 0:
                print(
                    f'{type(self.base_ensemble_).__name__} already trained, '
                    f'ignoring n_estimators and '
                    f'tree_max_depth parameters.')
        except NotFittedError:
            self.base_ensemble_ = self._validate_and_create_base_ensemble()
            if self.verbose > 0:
                print(
                    f'Training {type(self.base_ensemble_).__name__} '
                    f'base ensemble...')
            start_time = time.time()
            self.base_ensemble_.fit(X, y)
            end_time = time.time()
            self.ensemble_training_time_ = end_time - start_time
            if self.verbose > 0:
                print(
                    f'Finish training {type(self.base_ensemble_).__name__} '
                    f'base ensemble'
                    f' in {self.ensemble_training_time_} seconds.')

        start_time = time.time()

        # First step is extract the rules
        self._rule_extractor = RuleExtractorFactory.get_rule_extractor(
            self.base_ensemble_, self.column_names,
            self.classes_, self.X_, self.y_, self.float_threshold)
        if self.verbose > 0:
            print(
                f'Extracting rules from {type(self.base_ensemble_).__name__} '
                f'base ensemble...')
        self.original_rulesets_, \
        self._global_condition_map = self._rule_extractor.extract_rules()
        processed_rulesets = copy.deepcopy(self.original_rulesets_)

        # We create the heuristics object which will compute all the
        # heuristics related measures
        self._rule_heuristics = RuleHeuristics(X=self.X_, y=self.y_,
                                               classes_=self.classes_,
                                               condition_map=
                                               self._global_condition_map,
                                               cov_threshold=self.cov_threshold,
                                               conf_threshold=
                                               self.conf_threshold)
        if self.verbose > 0:
            print(f'Initializing sets and computing condition map...')
        self._initialize_sets()

        if str(
                self.base_ensemble_.__class__) == \
                "<class 'catboost.core.CatBoostClassifier'>":
            for ruleset in processed_rulesets:
                for rule in ruleset:
                    new_A = self._remove_opposite_conditions(set(rule.A),
                                                             rule.class_index)
                    rule.A = new_A

        for ruleset in processed_rulesets:
            self._rule_heuristics.compute_rule_heuristics(
                ruleset, recompute=True)
        self.simplified_ruleset_ = processed_rulesets[0]

        self._simplify_rulesets(
            self.simplified_ruleset_)
        y_pred = self._add_default_rule(self.simplified_ruleset_)
        self.simplified_ruleset_.compute_class_perf_fast(y_pred,
                                                         self.y_,
                                                         self.metric)
        self.simplified_ruleset_.rules.pop()

        self.n_combinations_ = 0

        self._early_stop_cnt = 0
        if self.early_stop > 0:
            early_stop = int(len(processed_rulesets) * self.early_stop)
        else:
            early_stop = len(processed_rulesets)

        if self.verbose > 0:
            print(f'Start combination process...')
            if self.verbose > 1:
                print(
                    f'Iteration {0}, Rule size: '
                    f'{len(self.simplified_ruleset_.rules)}, '
                    f'{self.metric}: '
                    f'{self.simplified_ruleset_.metric(self.metric)}')
        for i in range(1, len(processed_rulesets)):
            # combine the rules
            combined_rules = self._combine_rulesets(self.simplified_ruleset_,
                                                    processed_rulesets[i])

            if self.verbose > 1:
                print(f'Iteration{i}:')
                print(
                    f'\tCombined rules size: {len(combined_rules.rules)} rules')
            # prune inaccurate rules
            self._sequential_covering_pruning(combined_rules)
            if self.verbose > 1:
                print(
                    f'\tSequential covering pruned rules size: '
                    f'{len(combined_rules.rules)} rules')
            # simplify rules
            self._simplify_rulesets(combined_rules)
            if self.verbose > 1:
                print(
                    f'\tSimplified rules size: '
                    f'{len(combined_rules.rules)} rules')

            # skip if the combined rules are empty
            if len(combined_rules.rules) == 0:
                if self.verbose > 1:
                    print(f'\tCombined rules are empty, skipping iteration.')
                continue
            self.simplified_ruleset_, best_ruleset = self._evaluate_combinations(
                self.simplified_ruleset_, combined_rules)

            if self._early_stop_cnt >= early_stop:
                break
            if self.simplified_ruleset_.metric() == 1:
                break

        self.simplified_ruleset_.rules[:] = [rule for rule in
                                             self.simplified_ruleset_.rules
                                             if rule.cov > 0]
        if self.verbose > 0:
            print(f'Finish combination process, adding default rule...')

        _ = self._add_default_rule(self.simplified_ruleset_)
        self.simplified_ruleset_.prune_condition_map()
        end_time = time.time()
        self.combination_time_ = end_time - start_time
        if self.verbose > 0:
            print(
                f'R size: {len(self.simplified_ruleset_.rules)}, {self.metric}:'
                f' {self.simplified_ruleset_.metric(self.metric)}')

        return self

    def _more_tags(self):
        return {'binary_only': True}

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        if self.X_.shape[1] != X.shape[1]:
            raise ValueError(
                f"X contains {X.shape[1]} features, but RuleGenExClassifier was"
                f" fitted with {self.X_.shape[1]}."
            )

        return self.simplified_ruleset_.predict(X)

    def predict_proba(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        if self.X_.shape[1] != X.shape[1]:
            raise ValueError(
                f"X contains {X.shape[1]} features, but RuleGenExClassifier was"
                f" fitted with {self.X_.shape[1]}."
            )
        return self.simplified_ruleset_.predict_proba(X)

    def _initialize_sets(self):
        self._bad_combinations = set()
        self._good_combinations = dict()
        self._rule_heuristics.initialize_sets()

    def _validate_and_create_base_ensemble(self):

        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be greater than zero, "
                             "got {0}.".format(self.n_estimators))
        if self.base_ensemble is None:
            if is_classifier(self):
                self.base_ensemble_ = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            elif is_regressor(self):
                self.base_ensemble_ = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            else:
                raise ValueError(
                    "You should choose an original classifier/regressor "
                    "ensemble to use RuleGenEx method.")
        self.base_ensemble_.n_estimators = self.n_estimators
        if str(
                self.base_ensemble_.__class__) == \
                "<class 'catboost.core.CatBoostClassifier'>":
            self.base_ensemble_.set_params(n_estimators=self.n_estimators,
                                           depth=self.tree_max_depth)
        elif isinstance(self.base_ensemble_, BaggingClassifier):
            if is_classifier(self):
                self.base_ensemble_.base_estimator = DecisionTreeClassifier(
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
            else:
                self.base_ensemble_.base_estimator = DecisionTreeRegressor(
                    max_depth=self.tree_max_depth,
                    random_state=self.random_state)
        else:
            self.base_ensemble_.max_depth = self.tree_max_depth
        return clone(self.base_ensemble_)

    def _remove_opposite_conditions(self, conditions, class_index):
        att_op_list = [[cond[1].att_index,
                        cond[1].op.__name__, cond[0]]
                       for cond in conditions]

        att_op_list = np.array(att_op_list, dtype=object)
        att_op_list = att_op_list[att_op_list[:, 0].argsort()]

        # Second part is to remove opposite operator
        # conditions (e.g. att1>=5  att1<5)
        # create list for removing opposite conditions
        dict_opp_cond = {
            i[0]: att_op_list[(att_op_list == i[0]).nonzero()[0]][:, 2]
            for i in att_op_list}
        # create generator to traverse just conditions with the same att_index
        # and different operator that appear
        # more than once
        gen_opp_cond = ((att, conds) for (att, conds) in dict_opp_cond.items()
                        if len(conds) > 1)
        for (_, conds) in gen_opp_cond:
            list_conds = [(int(id_),
                           self._rule_heuristics.get_conditions_heuristics(
                               {(int(id_),
                                 self._global_condition_map[int(id_)])})[0][
                               'supp'][class_index])
                          for id_
                          in conds]
            best_condition = max(list_conds, key=lambda item: item[1])
            list_conds.remove(
                best_condition)  # remove the edge condition of the box from
            # the list so it will remain
            [conditions.remove((cond[0], self._global_condition_map[cond[0]]))
             for cond in list_conds]
        return frozenset(conditions)

    def _sort_ruleset(self, ruleset):
        if len(ruleset.rules) == 0:
            return

        ruleset.rules.sort(key=lambda rule: (-1 * len(rule.A), rule.str))
        for attr in self._sorting_list:
            ruleset.rules.sort(key=attrgetter(attr), reverse=True)

        if self.sort_by_class is not None:
            ruleset.rules.sort(key=lambda rule: self.sort_by_class.index(rule.y.item()))

    def _combine_rulesets(self, ruleset1, ruleset2):
        combined_rules = set()
        for class_one in self.classes_:
            for class_two in self.classes_:
                s_ruleset1 = [rule1 for rule1 in ruleset1 if
                              (rule1.y == [class_one])]
                s_ruleset2 = [rule2 for rule2 in ruleset2 if
                              (rule2.y == [class_two])]
                combined_rules.update(
                    self._combine_sliced_rulesets(s_ruleset1, s_ruleset2))
        combined_rules = RuleSet(list(combined_rules),
                                 self._global_condition_map,
                                 classes=self.classes_)
        self._sort_ruleset(combined_rules)
        return combined_rules

    def _combine_sliced_rulesets(self, s_ruleset1, s_ruleset2):
        combined_rules = set()

        for r1 in s_ruleset1:
            for r2 in s_ruleset2:
                if len(r1.A) == 0 or len(r2.A) == 0:
                    continue
                heuristics_dict = self._rule_heuristics.combine_heuristics(
                    r1.heuristics_dict, r2.heuristics_dict)

                r1_AUr2_A = set(r1.A.union(r2.A))

                if heuristics_dict['cov'] == 0:
                    self._bad_combinations.add(frozenset(r1_AUr2_A))
                    continue

                if frozenset(r1_AUr2_A) in self._bad_combinations:
                    continue

                self.n_combinations_ += 1  

                weight = None

                if self._weights is None:
                    ens_class_dist = np.mean(
                        [r1.ens_class_dist, r2.ens_class_dist],
                        axis=0).reshape(
                        (len(self.classes_),))
                else:
                    ens_class_dist = np.average(
                        [r1.ens_class_dist, r2.ens_class_dist],
                        axis=0,
                        weights=[r1.weight,
                                 r2.weight]).reshape(
                        (len(self.classes_),))
                    weight = (r1.weight() + r2.weight) / 2
                logit_score = 0

                class_dist = ens_class_dist
                y_class_index = np.argmax(class_dist).item()
                y = np.array([self.classes_[y_class_index]])

                new_rule = Rule(frozenset(r1_AUr2_A),
                                class_dist=class_dist,
                                ens_class_dist=ens_class_dist,
                                local_class_dist=ens_class_dist,
                                # rule_class_dist,
                                logit_score=logit_score, y=y,
                                y_class_index=y_class_index,
                                classes=self.classes_, weight=weight)

                if new_rule in self._good_combinations:
                    heuristics_dict = self._good_combinations[new_rule]
                    new_rule.set_heuristics(heuristics_dict)
                    combined_rules.add(new_rule)
                else:
                    new_rule.set_heuristics(heuristics_dict)
                    if new_rule.conf > self.conf_threshold and \
                            new_rule.cov > self.cov_threshold:
                        combined_rules.add(new_rule)
                        self._good_combinations[new_rule] = heuristics_dict
                    else:
                        self._bad_combinations.add(frozenset(r1_AUr2_A))

        return combined_rules

    def _simplify_conditions(self, conditions):
        cond_map = self._global_condition_map  

        att_op_list = [
            [(cond[1].att_index, cond[1].op.__name__), cond[0]]
            for cond in conditions]
        att_op_list = np.array(att_op_list, dtype=object)

        dict_red_cond = {
            i[0]: att_op_list[(att_op_list == i[0]).nonzero()[0]][:, 1]
            for i in att_op_list}
        
        gen_red_cond = ((att_op, conds) for (att_op, conds) in
                        dict_red_cond.items() if len(conds) > 1)

        for (att_op, conds) in gen_red_cond:
            tup_att_op = literal_eval(att_op)
            list_conds = {cond_map[int(id_)] for id_ in conds}
            if tup_att_op[1] in ['lt', 'le']:
                edge_condition = min(list_conds, key=lambda
                    item: item.value)  # condition at the edge of the box
            if tup_att_op[1] in ['gt', 'ge']:
                edge_condition = max(list_conds, key=lambda item: item.value)
            list_conds.remove(
                edge_condition)  
            
            [conditions.remove((hash(cond), cond)) for cond in list_conds]

        return frozenset(conditions)

    def _sequential_covering_pruning(self, ruleset):
        return_ruleset = []
        not_cov_samples = self._rule_heuristics.ones
        found_rule = True
        while len(
                ruleset.rules) > 0 and \
                self._rule_heuristics.bitarray_.get_number_ones(
                    not_cov_samples) > 0 \
                and found_rule:
            self._rule_heuristics.compute_rule_heuristics(ruleset,
                                                          not_cov_samples)
            self._sort_ruleset(ruleset)
            found_rule = False
            for rule in ruleset:
                result, \
                not_cov_samples = self._rule_heuristics.rule_is_accurate(
                    rule,
                    not_cov_samples=not_cov_samples)
                if result:
                    return_ruleset.append(rule)
                    ruleset.rules.remove(rule)
                    found_rule = True
                    break
        ruleset.rules[:] = return_ruleset

    def _simplify_rulesets(self, ruleset):
        for rule in ruleset:
            rule.A = self._simplify_conditions(set(rule.A))
            rule.update_string_representation()
            base_line_error = self._compute_pessimistic_error(rule.A,
                                                              rule.class_index)
            min_error = 0
            while min_error <= base_line_error and len(rule.A) > 0 \
                    and base_line_error > 0:
                errors = [(cond,
                           self._compute_pessimistic_error(
                               rule.A.difference({cond}), rule.class_index),
                           cond[1].str)
                          for cond
                          in rule.A]

                min_error_tup = min(errors, key=lambda tup: (tup[1],
                                                             tup[2]))

                min_error = min_error_tup[1]
                if min_error <= base_line_error:
                    base_line_error = min_error
                    min_error = 0
                    rule_conds = set(rule.A)
                    rule_conds.remove(min_error_tup[0])
                    rule.A = frozenset(rule_conds)
                    rule.update_string_representation()

        self._rule_heuristics.compute_rule_heuristics(ruleset, recompute=True)

        ruleset.rules[:] = [rule for rule in ruleset
                            if 0 < len(rule.A)
                            and rule.cov > self.cov_threshold
                            and rule.conf > self.conf_threshold
                            ]

        self._rule_heuristics.compute_rule_heuristics(ruleset,
                                                      sequential_covering=True)
        self._sort_ruleset(ruleset)

    def _compute_pessimistic_error(self, conditions, class_index,
                                   not_cov_samples=None):

        if len(conditions) == 0:
            e = (self.X_.shape[
                     0] - self._rule_heuristics.bitarray_.get_number_ones(
                self._rule_heuristics.training_bit_sets[
                    class_index])) / self.X_.shape[0]
            return 100 * _statistical_error_estimate(self.X_.shape[0], e,
                                                     self.alpha_half_)

        heuristics_dict, _ = self._rule_heuristics.get_conditions_heuristics(
            conditions, not_cov_mask=not_cov_samples)

        total_instances = heuristics_dict['cov_count']
        accurate_instances = heuristics_dict['class_cov_count'][class_index]
        error_instances = total_instances - accurate_instances

        if total_instances == 0:
            return 0
        else:
            e = error_instances / total_instances  # totalInstances

        return 100 * _statistical_error_estimate(total_instances, e,
                                                 self.alpha_half_)

    def _add_default_rule(self, ruleset):

        predictions, covered_instances = ruleset._predict(self.X_)
        not_cov_samples = ~covered_instances

        all_covered = False
        if not_cov_samples.sum() == 0:
            uncovered_dist = np.array(
                [self._rule_heuristics.bitarray_.get_number_ones(
                    self._rule_heuristics.training_bit_sets[i]) for i in
                    range(len(self.classes_))])
            all_covered = True
        else:
            uncovered_labels = self.y_[not_cov_samples]
            uncovered_dist = np.array(
                [(uncovered_labels == class_).sum() for class_ in
                 self.classes_])

        default_class_idx = np.argmax(uncovered_dist)
        predictions[not_cov_samples] = self.classes_[default_class_idx]
        default_rule = Rule({},
                            class_dist=uncovered_dist / uncovered_dist.sum(),
                            y=np.array([self.classes_[default_class_idx]]),
                            y_class_index=default_class_idx,
                            classes=self.classes_, n_samples=uncovered_dist)
        if not all_covered:
            default_rule.cov = not_cov_samples.sum() / self.X_.shape[0]
            default_rule.conf = uncovered_dist[
                                    default_class_idx] / not_cov_samples.sum()
            default_rule.supp = uncovered_dist[default_class_idx] / \
                                self.X_.shape[0]
        ruleset.rules.append(default_rule)

        return predictions

    def _evaluate_combinations(self, simplified_ruleset, combined_rules):
        y_pred = self._add_default_rule(combined_rules)

        combined_rules.compute_class_perf_fast(y_pred, self.y_,
                                               self.metric)

        # if rule_added:
        combined_rules.rules.pop()

        if combined_rules.metric(self.metric) > simplified_ruleset.metric(
                self.metric):
            self._early_stop_cnt = 0
            if self.verbose > 1:
                print(
                    f'\tBest {self.metric}, Combined rules: '
                    f'{combined_rules.metric(self.metric)}')
            return combined_rules, 'comb'
        else:
            self._early_stop_cnt += 1
            if self.verbose > 1:
                print(
                    f'\tBest {self.metric}, Previous combined rules: '
                    f'{simplified_ruleset.metric(self.metric)}')
            return simplified_ruleset, 'simp'
