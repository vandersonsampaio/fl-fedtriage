import numpy as np
import pytest

from src.defense.adaaggrl_agent import compute_weights_and_penalty
from src.triage import TriageLearner, TriageRule
from src.triage.rule import minmax
from src.triage.signals import INVERSION, MEDIAN_COSINE, SERVER_COSINE, median_cosine, server_cosine
from tests.fixtures.mock_data import make_synthetic_participants


class TestSignals:
    def test_server_cosine_is_low_for_opposite_update(self):
        srv = np.array([1.0, 1.0, 0.0])
        cos = server_cosine([srv * 2, -srv, np.array([0.0, 0.0, 1.0])], srv)
        np.testing.assert_allclose(cos, [1.0, -1.0, 0.0], atol=1e-9)

    def test_median_cosine_is_low_for_minority_outlier(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal(8)
        updates = [base + rng.standard_normal(8) * 0.05 for _ in range(5)] + [-base]
        cos = median_cosine(updates)
        assert cos.argmin() == 5
        assert cos[:5].min() > 0.9


class TestTriageRule:
    def test_threshold_is_fraction_of_round_maximum(self):
        d = TriageRule([SERVER_COSINE], b=0.5).decide({SERVER_COSINE: np.array([0.0, 0.4, 0.6, 1.0])})
        assert d.thresholds[SERVER_COSINE] == pytest.approx(0.5)
        np.testing.assert_array_equal(d.included, [False, False, True, True])

    def test_uniform_weights_ignore_signal_magnitude(self):
        d = TriageRule([SERVER_COSINE]).decide({SERVER_COSINE: np.array([-1.0, 0.7, 0.8, 1.0])})
        np.testing.assert_allclose(d.weights, [0.0, 1 / 3, 1 / 3, 1 / 3])

    def test_samples_weighting(self):
        d = TriageRule([SERVER_COSINE], weighting="samples").decide(
            {SERVER_COSINE: np.array([-1.0, 0.9, 1.0])}, sample_sizes=[100, 10, 30])
        np.testing.assert_allclose(d.weights, [0.0, 0.25, 0.75])

    def test_samples_weighting_requires_sizes(self):
        with pytest.raises(ValueError):
            TriageRule([SERVER_COSINE], weighting="samples").decide({SERVER_COSINE: np.ones(3)})

    def test_any_and_all_combiners(self):
        raw = {INVERSION: np.array([0.0, 1.0, 1.0, 0.9]), SERVER_COSINE: np.array([1.0, 0.0, 1.0, 0.9])}
        any_ = TriageRule([INVERSION, SERVER_COSINE], combine="any").decide(raw)
        all_ = TriageRule([INVERSION, SERVER_COSINE], combine="all").decide(raw)
        np.testing.assert_array_equal(any_.included, [False, False, True, True])
        np.testing.assert_array_equal(all_.included, [True, True, True, True])

    def test_confidence_uses_the_signal_with_the_largest_gap(self):
        raw = {INVERSION: np.array([0.0, 0.1, 0.9, 1.0]), SERVER_COSINE: np.array([0.0, 0.4, 0.6, 1.0])}
        d = TriageRule([INVERSION, SERVER_COSINE], combine="confidence").decide(raw)
        np.testing.assert_array_equal(d.included, [False, False, True, True])
        assert d.reasons() == ["excluded:inversion", "excluded:inversion", "included", "included"]

    def test_min_rank_flags_the_lowest_rank_of_any_signal(self):
        raw = {INVERSION: np.array([0.1, 5.0, 4.0, 3.0, 2.0]), SERVER_COSINE: np.array([0.9, 0.1, 0.8, 0.7, 0.6])}
        d = TriageRule([INVERSION, SERVER_COSINE], b=0.25, combine="min_rank").decide(raw)
        np.testing.assert_array_equal(d.included, [False, False, True, True, True])
        assert d.reasons()[:2] == ["excluded:inversion", "excluded:server_cosine"]

    def test_mad_threshold_flags_only_outliers(self):
        raw = {SERVER_COSINE: np.array([-0.9, 0.80, 0.82, 0.85, 0.86, 0.88, 0.9, 0.91, 0.93, 0.95])}
        mad = TriageRule([SERVER_COSINE], threshold="mad", k=3.0).decide(raw)
        rel = TriageRule([SERVER_COSINE], b=0.5).decide(raw)
        np.testing.assert_array_equal(mad.included, [False] + [True] * 9)
        np.testing.assert_array_equal(rel.included, mad.included)
        no_attack = {SERVER_COSINE: np.array([0.80, 0.82, 0.85, 0.86, 0.88, 0.9, 0.91, 0.93, 0.95, 0.96])}
        assert TriageRule([SERVER_COSINE], threshold="mad").decide(no_attack).included.all()
        assert not TriageRule([SERVER_COSINE], b=0.5).decide(no_attack).included.all()

    def test_memory_counter_and_penalty(self):
        rule = TriageRule([SERVER_COSINE], lam=2.0)
        d1 = rule.decide({SERVER_COSINE: np.array([0.0, 1.0, 1.0])})
        np.testing.assert_array_equal(d1.h, [1, 0, 0])
        d2 = rule.decide({SERVER_COSINE: np.array([1.0, 1.0, 0.0])}, h=d1.h)
        np.testing.assert_array_equal(d2.h, [0, 0, 1])
        d3 = rule.decide({SERVER_COSINE: np.array([1.0, 1.0, 0.0])}, h=np.array([2.0, 0.0, 0.0]))
        np.testing.assert_allclose(d3.weights, [1 / 3, 2 / 3, 0.0])

    def test_all_excluded_falls_back_to_uniform(self):
        d = TriageRule([SERVER_COSINE], b=1.0).decide({SERVER_COSINE: np.array([0.3, 0.3, 0.3])})
        assert d.fallback
        np.testing.assert_allclose(d.weights, np.ones(3) / 3)
        assert d.reasons() == ["fallback"] * 3

    def test_soft_weighting_matches_adaaggrl_formula(self):
        rng = np.random.default_rng(1)
        score, h = rng.standard_normal(10), rng.integers(0, 3, 10).astype(float)
        d = TriageRule([INVERSION], b=0.25, weighting="soft", lam=2.0).decide({INVERSION: score}, h=h)
        ref_w, ref_h, _ = compute_weights_and_penalty(
            np.column_stack([score, np.zeros((10, 3))]), np.array([1.0, 0, 0, 0, 0.25]), h, lam=2.0)
        np.testing.assert_array_equal(d.weights, ref_w / ref_w.sum())
        np.testing.assert_array_equal(d.h, ref_h)

    def test_records_name_the_deciding_signal(self):
        raw = {INVERSION: np.array([0.0, 1.0, 1.0]), SERVER_COSINE: np.array([0.0, 0.0, 1.0])}
        d = TriageRule([INVERSION, SERVER_COSINE]).decide(raw)
        rows = d.records(3, ["a", "b", "c"])
        assert [r["reason"] for r in rows] == ["excluded:inversion+server_cosine", "excluded:server_cosine", "included"]
        assert rows[1]["round"] == 3 and rows[1]["client"] == "b"
        assert rows[1][f"{SERVER_COSINE}_flag"] and not rows[1][f"{INVERSION}_flag"]
        assert rows[2]["weight"] == pytest.approx(1.0)
        assert minmax(raw[INVERSION]).tolist() == pytest.approx([r[f"{INVERSION}_norm"] for r in rows])

    @pytest.mark.parametrize("kwargs", [
        {"signals": []}, {"signals": ["norm"]}, {"signals": [INVERSION], "combine": "majority"},
        {"signals": [INVERSION], "weighting": "trust"}, {"signals": [INVERSION, SERVER_COSINE], "weighting": "soft"},
        {"signals": [INVERSION], "lam": 0.5}, {"signals": [INVERSION], "threshold": "quantile"},
    ])
    def test_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            TriageRule(**kwargs)


class TestTriageLearner:
    def _train(self, attack_type="sign_flipping", seed=0, **rule_kwargs):
        participants = make_synthetic_participants()
        learner = TriageLearner(n_rounds=3, n_classes=2, attack_type=attack_type, byzantine_ids=[0, 1],
                                seed=seed, rule=TriageRule(**rule_kwargs))
        history = learner.train(participants, verbose=False)
        return learner, history

    def test_logs_one_record_per_client_and_round(self):
        learner, history = self._train(signals=[SERVER_COSINE, MEDIAN_COSINE])
        assert len(history) == 3
        assert len(learner.log) == 3 * 10
        assert {r["round"] for r in learner.log} == {1, 2, 3}

    def test_excludes_sign_flipping_attackers(self):
        learner, history = self._train(signals=[SERVER_COSINE])
        byz = {"client_0", "client_1"}
        assert all(not r["included"] for r in learner.log if r["client"] in byz)
        assert history[-1].global_accuracy > 0.8

    def test_same_seed_same_run(self):
        a, ha = self._train(signals=[SERVER_COSINE, MEDIAN_COSINE], seed=5)
        b, hb = self._train(signals=[SERVER_COSINE, MEDIAN_COSINE], seed=5)
        assert [r.global_accuracy for r in ha] == [r.global_accuracy for r in hb]
        np.testing.assert_array_equal(a.global_params, b.global_params)

    def test_inversion_requires_logistic_model(self):
        with pytest.raises(ValueError):
            TriageLearner(n_rounds=1, n_classes=10, model_type="cnn", input_shape=(28, 28, 1),
                          rule=TriageRule([INVERSION]))
