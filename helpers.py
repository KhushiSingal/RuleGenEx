from rulegenex.rules import RuleSet, Rule, Condition


def total_n_rules(list_of_rulesets):
    return sum([len(ruleset.get_rule_list()) for ruleset in list_of_rulesets])


def count_keys(dict_, key):
    return (key in dict_) + sum(
        count_keys(v, key) for v in dict_.values() if isinstance(v, dict))



