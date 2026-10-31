import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score

from src.experiment_protocol_v2 import (ClusterMetricCache, operating_metrics,
                                       patient_bootstrap, threshold_options, validation_score)


def test_alarm_cap_changes_operating_threshold():
    y = np.r_[np.ones(10), np.zeros(100)]
    score = np.r_[np.full(5, .9), np.full(5, .1), np.full(5, .8), np.full(95, .01)]
    choices = threshold_options(y, score, .03)
    assert choices['selected']['threshold'] == .9
    assert choices['selected']['false_positive_rate'] == 0
    assert choices['unrestricted_max_f1']['threshold'] == .1
    assert choices['unrestricted_max_f1']['false_positive_rate'] == .05


def test_thresholds_include_low_scores_and_group_ties():
    choices = threshold_options([0, 1, 1], [.001, .01, .01])
    assert choices['selected']['threshold'] == .01
    assert choices['selected']['f1'] == 1
    assert choices['selected']['tp'] == 2


def test_guardrails_and_multi_metric_score():
    y = np.r_[np.ones(20), np.zeros(100)]
    score = np.r_[np.full(10, .9), np.full(10, .1), np.full(2, .8), np.full(98, .01)]
    reference = operating_metrics(y, score, .8)
    # Use a reference with >=95% accuracy for eligibility.
    reference.update(accuracy=.97)
    assert np.isclose(validation_score(reference, reference), 1.)
    bad = {**reference, 'false_positive_rate': .031}
    assert validation_score(bad, reference) is None
    assert validation_score({**reference, 'auroc': reference['auroc']-.006}, reference) is None


def test_cluster_metrics_match_explicit_patient_resampling_with_ties():
    y = np.array([0, 1, 0, 1, 1, 0, 0])
    score = np.array([.2, .8, .8, .6, .6, .1, .3])
    lengths = np.array([2, 3, 2])
    counts = np.array([2, 1, 0])
    cache = ClusterMetricCache(y, score, .6, lengths)
    actual = cache.calculate(counts)
    indexes = np.r_[0, 1, 0, 1, 2, 3, 4]
    expected = operating_metrics(y[indexes], score[indexes], .6)
    for key in actual:
        assert actual[key] == pytest.approx(expected[key])


def test_zero_weight_tie_groups_are_safe():
    y = np.array([0, 1, 0, 1])
    score = np.array([.99, .95, .1, .8])
    value = ClusterMetricCache(y, score, .5, [2, 2]).calculate([0, 2])
    assert value['auroc'] == 1
    assert value['auprc'] == 1


def test_patient_bootstrap_is_paired_and_repeatable():
    y = np.array([0, 1, 0, 1, 0, 1])
    score = np.array([.1, .9, .2, .8, .3, .7])
    prediction = {'selected': (score, .5), 'reference': (score, .5)}
    a = patient_bootstrap(y, prediction, [2, 2, 2], 20, seed=7)
    b = patient_bootstrap(y, prediction, [2, 2, 2], 20, seed=7)
    assert a == b
    assert a['valid_replicates'] == 20
    for v in a['paired_differences']['selected_minus_reference'].values():
        assert v == {'lower': 0., 'upper': 0.}


def test_bad_grouping_and_single_class_rejected():
    with pytest.raises(ValueError):
        ClusterMetricCache([0, 1], [.1, .9], .5, [1])
    with pytest.raises(ValueError):
        threshold_options([0, 0], [.1, .9])


def test_ranking_does_not_change_when_threshold_changes():
    y = [0, 1, 0, 1]
    score = [.1, .8, .7, .9]
    a = operating_metrics(y, score, .5)
    b = operating_metrics(y, score, .85)
    assert a['auroc'] == b['auroc']
    assert a['auprc'] == b['auprc']
    assert a['recall'] != b['recall']
