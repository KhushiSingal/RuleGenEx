import json
import copy
import operator
from os import path
from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from tempfile import TemporaryDirectory
from math import copysign

from scipy.special import expit, logsumexp, softmax
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from .helpers import count_keys
from .rules import RuleSet, Condition, Rule


class BaseRuleExtractor(metaclass=ABCMeta):
    def __init__(self, _ensemble, _column_names, classes_, X, y,
                 float_threshold):
        self._column_names = _column_names
        self.classes_ = classes_
        self._ensemble = _ensemble
        self.X = X
        self.y = y
        self.float_threshold = float_threshold
        _, counts = np.unique(self.y, return_counts=True)
        self.class_ratio = counts.min() / counts.max()

    def get_tree_dict(self, base_tree, n_nodes=0):
        return {'children_left': base_tree.tree_.children_left,
                'children_right': base_tree.tree_.children_right,
                'feature': base_tree.tree_.feature,
                'threshold': base_tree.tree_.threshold,
                'value': base_tree.tree_.value,
                'n_samples': base_tree.tree_.weighted_n_node_samples,
                'n_nodes': base_tree.tree_.node_count}

    @abstractmethod
    def create_new_rule(self, node_index, tree_dict, condition_set=None,
                        logit_score=None, weights=None,
                        tree_index=None, class_index=None):
        """ Creates a new rule with all the information in the parameters

        :param node_index: the index of the leaf node

        :param tree_dict: a dictionary containing  the information of the
            base_tree (arrays on:class: `sklearn.tree.Tree` class

        :param condition_set: set of :class:`rulegenex.rule.Condition` objects
            of the new rule

        :param logit_score: logit_score of the rule (only applies for Gradient
            Boosting Trees)

        :param weights: weight of the new rule

        :param tree_index: index of the tree inside the ensemble

        :param class_index: index of the class that this rule predicts

        :return: a :class:`rulegenex.rules.Rule` object
        """

    @abstractmethod
    def extract_rules(self):
        """ Main method for extracting the rules of tree ensembles

        :return: an array of :class:`rulegenex.rules.RuleSet'
        """

    def get_base_ruleset(self, tree_dict, class_index=None, condition_map=None,
                         tree_index=None):

        if condition_map is None:
            condition_map = dict()  # dictionary of conditions A

        extracted_rules = self.recursive_extraction(tree_dict, tree_index,
                                                    class_index=class_index,
                                                    node_index=0,
                                                    condition_map=condition_map,
                                                    condition_set=set())
        return RuleSet(extracted_rules, condition_map, classes=self.classes_)

    def recursive_extraction(self, tree_dict, tree_index=0, class_index=None,
                             node_index=0,
                             condition_map=None, condition_set=None):
        if condition_map is None:
            condition_map = dict()
        if condition_set is None:
            condition_set = set()
        rules = []
        children_left = tree_dict['children_left']
        children_right = tree_dict['children_right']
        feature = tree_dict['feature']
        threshold = tree_dict['threshold']

        # leaf node so a rule is created
        if children_left[node_index] == children_right[node_index]:
            weights = None
            logit_score = None
            new_rule = self.create_new_rule(node_index, tree_dict,
                                            condition_set, logit_score, weights,
                                            tree_index, class_index=class_index)
            rules.append(new_rule)
        else:
            # create condition, add it to the condition_set and get conditions from left and right child
            att_name = None
            if self._column_names is not None:
                att_name = self._column_names[feature[node_index]]
            condition_set_left = copy.copy(condition_set)
            # condition_set_left = copy.copy(condition_set)
            # determine operators
            op_left, op_right = self.get_split_operators()

            # -0 problem solution
            split_value = threshold[node_index]
            if abs(split_value) < self.float_threshold:
                split_value = copysign(self.float_threshold, split_value)
                # split_value=0
                # print(split_value)
            new_condition_left = Condition(feature[node_index], op_left,
                                           # threshold[node_index],
                                           split_value,
                                           att_name)
            condition_map[hash(new_condition_left)] = new_condition_left
            # condition_set_left.add(hash(new_condition_left))
            condition_set_left.add(
                (hash(new_condition_left), new_condition_left))
            left_rules = self.recursive_extraction(tree_dict, tree_index,
                                                   node_index=children_left[
                                                       node_index],
                                                   class_index=class_index,
                                                   condition_set=condition_set_left,
                                                   condition_map=condition_map)
            rules = rules + left_rules

            condition_set_right = copy.copy(condition_set)
            # condition_set_right = copy.copy(condition_set)
            new_condition_right = Condition(feature[node_index], op_right,
                                            # threshold[node_index],
                                            split_value,
                                            att_name)
            condition_map[hash(new_condition_right)] = new_condition_right
            # condition_set_right.add(hash(new_condition_right))
            condition_set_right.add(
                (hash(new_condition_right), new_condition_right))
            right_rules = self.recursive_extraction(tree_dict, tree_index,
                                                    class_index=class_index,
                                                    node_index=children_right[
                                                        node_index],
                                                    condition_set=condition_set_right,
                                                    condition_map=condition_map)
            rules = rules + right_rules
        return rules

    def get_split_operators(self):
        op_left = operator.le  # Operator.LESS_OR_EQUAL_THAN
        op_right = operator.gt  # Operator.GREATER_THAN
        return op_left, op_right


class DecisionTreeRuleExtractor(BaseRuleExtractor):
    def extract_rules(self):
        global_condition_map = dict()
        original_ruleset = self.get_base_ruleset(
            self.get_tree_dict(self._ensemble))
        global_condition_map.update(original_ruleset.condition_map)
        return original_ruleset, global_condition_map

    def create_new_rule(self, node_index, tree_dict, condition_set=None,
                        logit_score=None, weights=None,
                        tree_index=None, class_index=None):
        if condition_set is None:
            condition_set = {}
        value = tree_dict['value']
        n_samples = tree_dict['n_samples']

        if weights is not None:
            weight = weights[tree_index]
        else:
            weight = None
        class_dist = (value[node_index] / value[node_index].sum()).reshape(
            (len(self.classes_),))
        # predict y_class_index = np.argmax(class_dist).item()
        y_class_index = np.argmax(class_dist)
        y = np.array([self.classes_[y_class_index]])

        return Rule(frozenset(condition_set), class_dist=class_dist,
                    ens_class_dist=class_dist,
                    logit_score=logit_score, y=y,
                    y_class_index=y_class_index,
                    n_samples=n_samples[node_index], classes=self.classes_,
                    weight=weight)


def get_class_dist(raw_to_proba):
    return np.array([1 - raw_to_proba.item(), raw_to_proba.item()])


class ClassifierRuleExtractor(BaseRuleExtractor):
    def extract_rules(self):
        rulesets = []
        global_condition_map = dict()

        for base_tree in self._ensemble:
            original_ruleset = self.get_base_ruleset(
                self.get_tree_dict(base_tree))
            rulesets.append(original_ruleset)
            global_condition_map.update(original_ruleset.condition_map)
        return rulesets, global_condition_map

    def create_new_rule(self, node_index, tree_dict, condition_set=None,
                        logit_score=None, weights=None,
                        tree_index=None, class_index=None):
        if condition_set is None:
            condition_set = {}
        value = tree_dict['value']
        n_samples = tree_dict['n_samples']

        if weights is not None:
            weight = weights[tree_index]
        else:
            weight = None
        class_dist = (value[node_index] / value[node_index].sum()).reshape(
            (len(self.classes_),))
        # predict y_class_index = np.argmax(class_dist).item()
        y_class_index = np.argmax(class_dist)
        y = np.array([self.classes_[y_class_index]])

        return Rule(frozenset(condition_set), class_dist=class_dist,
                    ens_class_dist=class_dist,
                    logit_score=logit_score, y=y,
                    y_class_index=y_class_index,
                    n_samples=n_samples[node_index], classes=self.classes_,
                    weight=weight)


class GBMClassifierRuleExtractor(BaseRuleExtractor):
    def extract_rules(self):
        rulesets = []
        global_condition_map = dict()
        for tree_index, base_trees in enumerate(self._ensemble):
            original_ruleset = None
            for class_index, base_tree in enumerate(base_trees):
                if len(self.classes_) > 2:
                    if class_index == 0:
                        original_ruleset = self.get_base_ruleset(
                            self.get_tree_dict(base_tree),
                            class_index=class_index, tree_index=tree_index)
                        global_condition_map.update(
                            original_ruleset.condition_map)
                    else:
                        class_ruleset = self.get_base_ruleset(
                            self.get_tree_dict(base_tree),
                            class_index=class_index, tree_index=tree_index)
                        global_condition_map.update(
                            original_ruleset.condition_map)
                        original_ruleset.rules += class_ruleset.rules
                        original_ruleset.condition_map.update(
                            class_ruleset.condition_map)
                else:
                    original_ruleset = self.get_base_ruleset(
                        self.get_tree_dict(base_tree),
                        class_index=None, tree_index=tree_index)
            rulesets.append(original_ruleset)
            global_condition_map.update(original_ruleset.condition_map)
        return rulesets, global_condition_map

    def create_new_rule(self, node_index, tree_dict, condition_set=None,
                        logit_score=None, weights=None,
                        tree_index=None, class_index=None):
        if condition_set is None:
            condition_set = {}
        value = tree_dict['value']
        n_samples = tree_dict['n_samples']

        if tree_index == 0:
            init = self._get_gbm_init()
        else:
            init = np.zeros(value[node_index].shape)

        logit_score = init + value[node_index]

        if len(self.classes_) == 2:
            raw_to_proba = expit(logit_score)
            class_dist = get_class_dist(raw_to_proba)
        else:
            # raw_to_proba = logsumexp(logit_score)
            class_dist = np.zeros(len(self.classes_))
            if isinstance(logit_score, np.ndarray):
                class_dist[class_index] = logsumexp(logit_score)
            else:
                class_dist[class_index] = expit(logit_score)
                # class_dist[class_index] = expit(
                #     logit_score)  # logsumexp(logit_score)  # softmax(raw_to_proba)
            # class_dist = logit_score - logsumexp(logit_score)

        # predict y_class_index = np.argmax(class_dist).item()
        # y_class_index = np.argmax(class_dist).item()
        y_class_index = np.argmax(class_dist)
        y = np.array([self.classes_[y_class_index]])
        return Rule(frozenset(condition_set), class_dist=class_dist,
                    ens_class_dist=class_dist,
                    logit_score=logit_score, y=y,
                    y_class_index=y_class_index,
                    n_samples=n_samples[node_index], classes=self.classes_,
                    weight=weights)

    def _get_gbm_init(self):
        return self._ensemble._raw_predict_init(self.X[0].reshape(1, -1))


class XGBClassifierExtractor(GBMClassifierRuleExtractor):
    def extract_rules(self):
        rulesets = []
        global_condition_map = dict()
        booster = self._ensemble.get_booster()
        booster.feature_names = self._column_names.to_list()
        xgb_tree_dicts = booster.get_dump(dump_format='json')
        n_nodes = booster.trees_to_dataframe()[['Tree', 'Node']].groupby(
            'Tree').count().to_numpy()
        if len(self.classes_) > 2:
            t_idx = 0
            for tid in range(self._ensemble.n_estimators):
                current_tree_rules = []
                current_tree_condition_map = dict()
                for cid, _ in enumerate(self.classes_):
                    original_ruleset = self.get_base_ruleset(
                        self.get_tree_dict(xgb_tree_dicts[t_idx],
                                           n_nodes[t_idx]),
                        class_index=cid, tree_index=tid)
                    current_tree_rules.extend(original_ruleset.rules)
                    current_tree_condition_map.update(
                        original_ruleset.condition_map)
                    t_idx += 1
                current_tree_ruleset = RuleSet(current_tree_rules,
                                               current_tree_condition_map,
                                               classes=self.classes_)
                global_condition_map.update(current_tree_ruleset.condition_map)
                rulesets.append(current_tree_ruleset)
            return rulesets, global_condition_map

        else:  # binary classification

            for t_idx, xgb_t_dict in enumerate(xgb_tree_dicts):
                original_ruleset = self.get_base_ruleset(
                    self.get_tree_dict(xgb_t_dict, n_nodes[t_idx]),
                    class_index=0, tree_index=t_idx)
                rulesets.append(original_ruleset)
                global_condition_map.update(original_ruleset.condition_map)
            return rulesets, global_condition_map

    def get_tree_dict(self, base_tree, n_nodes=0):
        tree_dict = {'children_left': np.full(n_nodes, fill_value=-1),
                     'children_right': np.full(n_nodes, fill_value=-1),
                     'feature': np.full(n_nodes, fill_value=0),
                     'threshold': np.full(n_nodes, fill_value=0.0),
                     'value': np.full(n_nodes, fill_value=0.0),
                     'n_samples': np.full(n_nodes, fill_value=-1),
                     'n_nodes': n_nodes}

        tree = json.loads(base_tree)
        self._populate_tree_dict(tree, tree_dict)
        return tree_dict

    def _populate_tree_dict(self, tree, tree_dict):
        node_id = tree['nodeid']
        if 'leaf' in tree:
            tree_dict['value'][node_id] = tree['leaf']
            return
        if 'children' in tree:
            tree_dict['children_left'][node_id] = tree['children'][0]['nodeid']
            tree_dict['children_right'][node_id] = tree['children'][1]['nodeid']
            tree_dict['feature'][node_id] = \
                np.where(self._column_names == tree['split'])[
                    0].item()  
            tree_dict['threshold'][node_id] = tree['split_condition']
            self._populate_tree_dict(tree['children'][0], tree_dict)
            self._populate_tree_dict(tree['children'][1], tree_dict)

    def get_split_operators(self):
        op_left = operator.lt  
        op_right = operator.ge 
        return op_left, op_right

    def _get_gbm_init(self):
        if self._ensemble.base_score is None:
            return self.class_ratio
        else:
            return self._ensemble.base_score


class LGBMClassifierExtractor(GBMClassifierRuleExtractor):
    def extract_rules(self):
        rulesets = []
        global_condition_map = dict()
        booster = self._ensemble.booster_
        lgbm_tree_dicts = booster.dump_model()['tree_info']

        if len(self.classes_) > 2:
            t_idx = 0
            for tid in range(self._ensemble.n_estimators):
                current_tree_rules = []
                current_tree_condition_map = dict()
                for cid, _ in enumerate(self.classes_):
                    n_nodes = count_keys(lgbm_tree_dicts[t_idx],
                                         'split_index') + \
                              count_keys(lgbm_tree_dicts[t_idx], 'leaf_index')
                    original_ruleset = self.get_base_ruleset(
                        self.get_tree_dict(lgbm_tree_dicts[t_idx],
                                           n_nodes),
                        class_index=cid, tree_index=tid)
                    current_tree_rules.extend(original_ruleset.rules)
                    current_tree_condition_map.update(
                        original_ruleset.condition_map)
                    t_idx += 1
                current_tree_ruleset = RuleSet(current_tree_rules,
                                               current_tree_condition_map,
                                               classes=self.classes_)
                global_condition_map.update(current_tree_ruleset.condition_map)
                rulesets.append(current_tree_ruleset)
            return rulesets, global_condition_map

        else:  # binary classification
            for tree_index, lgbm_tree_dict in enumerate(lgbm_tree_dicts):
                n_nodes = count_keys(lgbm_tree_dict, 'split_index') + \
                          count_keys(lgbm_tree_dict, 'leaf_index')

                original_ruleset = self.get_base_ruleset(
                    self.get_tree_dict(lgbm_tree_dict, n_nodes),
                    class_index=0, tree_index=tree_index)
                rulesets.append(original_ruleset)
                global_condition_map.update(original_ruleset.condition_map)
            return rulesets, global_condition_map

    def get_tree_dict(self, base_tree, n_nodes=0):
        tree_dict = {'children_left': np.full(n_nodes, fill_value=-1),
                     'children_right': np.full(n_nodes, fill_value=-1),
                     'feature': np.full(n_nodes, fill_value=0),
                     'threshold': np.full(n_nodes, fill_value=0.0),
                     'value': np.full(n_nodes, fill_value=0.0),
                     'n_samples': np.full(n_nodes, fill_value=-1),
                     'n_nodes': n_nodes}

        self._populate_tree_dict(base_tree['tree_structure'], 0, 0, tree_dict)
        return tree_dict

    def _populate_tree_dict(self, tree, node_id, node_count, tree_dict):
        if 'leaf_value' in tree:
            tree_dict['value'][node_id] = tree['leaf_value']
            return node_count
        if 'left_child' in tree:
            tree_dict['feature'][node_id] = tree['split_feature']
            tree_dict['threshold'][node_id] = tree['threshold']

            node_count = node_count + 1
            l_id = node_count
            tree_dict['children_left'][node_id] = l_id
            node_count = self._populate_tree_dict(tree['left_child'], l_id,
                                                  node_count, tree_dict)

            node_count = node_count + 1
            r_id = node_count
            tree_dict['children_right'][node_id] = r_id
            node_count = self._populate_tree_dict(tree['right_child'], r_id,
                                                  node_count, tree_dict)
            return node_count

    def _get_gbm_init(self):
        return self.class_ratio


class CatBoostClassifierExtractor(GBMClassifierRuleExtractor):
    def __init__(self, _ensemble, _column_names, classes_, X, y,
                 float_threshold):
        super().__init__(_ensemble, _column_names, classes_, X, y,
                         float_threshold)
        self._splits = None
        self._leaf_nodes = None

    def extract_rules(self):
        rulesets = []
        global_condition_map = dict()
        with TemporaryDirectory() as tmp_dir_name:
            self._ensemble.save_model(path.join(tmp_dir_name, 'cat_tree.json'),
                                      format='json')
            cat_model = json.load(
                open(path.join(tmp_dir_name, 'cat_tree.json'), encoding='utf8'))
        cat_tree_dicts = cat_model['oblivious_trees']

        if len(self.classes_) > 2:
            for tid, cat_tree_dict in enumerate(cat_tree_dicts):
                current_tree_rules = []
                current_tree_condition_map = dict()
                tree_depth = len(cat_tree_dict['splits'])
                n_nodes = 2 ** (tree_depth + 1) - 1
                for cid, _ in enumerate(self.classes_):
                    original_ruleset = self.get_base_ruleset(
                        self.get_tree_dict(cat_tree_dict, n_nodes),
                        class_index=cid, tree_index=tid)
                    # remove rules with logit_score = 0
                    original_ruleset.rules[:] = [rule for rule in
                                                 original_ruleset.rules
                                                 if rule.logit_score != 0]
                    current_tree_rules.extend(original_ruleset.rules)
                    current_tree_condition_map.update(
                        original_ruleset.condition_map)
                current_tree_ruleset = RuleSet(current_tree_rules,
                                               current_tree_condition_map,
                                               classes=self.classes_)
                global_condition_map.update(current_tree_ruleset.condition_map)
                rulesets.append(current_tree_ruleset)
            return rulesets, global_condition_map
        else:
            for tree_index, cat_tree_dict in enumerate(cat_tree_dicts):
                tree_depth = len(cat_tree_dict['splits'])
                n_nodes = 2 ** (tree_depth + 1) - 1

                original_ruleset = self.get_base_ruleset(
                    self.get_tree_dict(cat_tree_dict, n_nodes),
                    tree_index=tree_index)
                # remove rules with logit_score = 0
                original_ruleset.rules[:] = [rule for rule in
                                             original_ruleset.rules
                                             if rule.logit_score != 0]
                rulesets.append(original_ruleset)
                global_condition_map.update(original_ruleset.condition_map)
            return rulesets, global_condition_map

    def get_tree_dict(self, base_tree, n_nodes=0):
        tree_dict = {'children_left': np.full(n_nodes, fill_value=-1),
                     'children_right': np.full(n_nodes, fill_value=-1),
                     'feature': np.full(n_nodes, fill_value=0),
                     'threshold': np.full(n_nodes, fill_value=0.0),
                     'value': np.full(n_nodes, fill_value=0.0),
                     'n_samples': np.full(n_nodes, fill_value=-1),
                     'n_nodes': n_nodes}

        self._splits = base_tree['splits']
        self._splits.reverse()
        self._leaf_nodes = base_tree['leaf_values']
        self._populate_tree_dict(base_tree, 0, 0, 0, tree_dict)
        return tree_dict

    def _populate_tree_dict(self, tree, node_id, node_count, tree_level,
                            tree_dict):
        if tree_level == len(self._splits):
            tree_dict['value'][node_id] = self._leaf_nodes.pop(0)
            return node_count
        else:
            tree_dict['feature'][node_id] = self._splits[tree_level][
                'float_feature_index']
            tree_dict['threshold'][node_id] = self._splits[tree_level]['border']

            tree_level = tree_level + 1

            node_count = node_count + 1
            l_id = node_count

            node_count = node_count + 1
            r_id = node_count

            tree_dict['children_left'][node_id] = l_id
            node_count = self._populate_tree_dict(tree, l_id, node_count,
                                                  tree_level, tree_dict)

            tree_dict['children_right'][node_id] = r_id
            node_count = self._populate_tree_dict(tree, r_id, node_count,
                                                  tree_level, tree_dict)
            return node_count

    def _get_gbm_init(self):
        return self.class_ratio


class RuleExtractorFactory:
    def get_rule_extractor(base_ensemble, column_names, classes, X, y,
                           float_threshold):
        if isinstance(base_ensemble, (
                AdaBoostClassifier, BaggingClassifier, RandomForestClassifier)):
            return ClassifierRuleExtractor(base_ensemble, column_names, classes,
                                           X, y, float_threshold)
        elif isinstance(base_ensemble, GradientBoostingClassifier):
            return GBMClassifierRuleExtractor(base_ensemble, column_names,
                                              classes, X, y, float_threshold)
        elif str(
                base_ensemble.__class__) == "<class 'xgboost.sklearn.XGBClassifier'>":
            return XGBClassifierExtractor(base_ensemble, column_names, classes,
                                          X, y, float_threshold)
        elif str(
                base_ensemble.__class__) == "<class 'lightgbm.sklearn.LGBMClassifier'>":
            return LGBMClassifierExtractor(base_ensemble, column_names, classes,
                                           X, y, float_threshold)
        elif str(
                base_ensemble.__class__) == "<class 'catboost.core.CatBoostClassifier'>":
            return CatBoostClassifierExtractor(base_ensemble, column_names,
                                               classes, X, y, float_threshold)
        elif isinstance(base_ensemble, DecisionTreeClassifier):
            return DecisionTreeRuleExtractor(base_ensemble, column_names,
                                             classes, X, y, float_threshold)
