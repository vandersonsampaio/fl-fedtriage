import numpy as np
import pytest

from src.classification.attack_simulator import (
    AttackSimulator, apply_trigger, backdoor_success_rate, image_shape,
)
from src.fl.attacked_learner import AttackedFederatedLearner
from tests.fixtures.mock_data import make_synthetic_participants


class TestTrigger:
    @pytest.mark.parametrize("shape", [(28, 28, 1), (28, 28, 3), (6, 6, 1)])
    def test_only_the_corner_patch_changes(self, shape):
        X = np.full((3, int(np.prod(shape))), 0.5, dtype=np.float32)
        out = apply_trigger(X, shape).reshape((3,) + shape)
        H, W, _ = shape
        patch = out[:, H - 5:H - 1, W - 5:W - 1, :]
        assert set(np.unique(patch)) == {0.0, 1.0}
        mask = np.ones(shape, dtype=bool)
        mask[H - 5:H - 1, W - 5:W - 1, :] = False
        assert np.all(out[:, mask] == 0.5)
        assert np.all(X == 0.5)

    def test_image_shape(self):
        assert image_shape(784) == (28, 28, 1)
        assert image_shape(2352, (28, 28, 3)) == (28, 28, 3)
        with pytest.raises(ValueError):
            image_shape(2352)
        with pytest.raises(ValueError):
            image_shape(784, (28, 28, 3))

    def test_poison_data_triggers_and_relabels_a_fraction(self):
        rng = np.random.default_rng(0)
        X = rng.random((40, 36)).astype(np.float32)
        y = rng.integers(1, 3, 40)
        Xp, yp = AttackSimulator(seed=0).poison_data(X, y, "backdoor_trigger", poison_fraction=0.5,
                                                     target_class=0)
        changed = np.any(Xp != X, axis=1)
        assert changed.sum() == 20
        assert np.all(yp[changed] == 0)
        assert np.all(yp[~changed] == y[~changed])
        np.testing.assert_array_equal(Xp[changed], apply_trigger(X[changed], (6, 6, 1)))

    def test_success_rate_ignores_target_class_samples(self):
        X = np.zeros((4, 36), dtype=np.float32)
        y = np.array([0, 1, 1, 2])
        always_target = lambda Z: np.zeros(len(Z), dtype=int)
        never_target = lambda Z: np.ones(len(Z), dtype=int)
        assert backdoor_success_rate(always_target, X, y, (6, 6, 1)) == 1.0
        assert backdoor_success_rate(never_target, X, y, (6, 6, 1)) == 0.0
        assert np.isnan(backdoor_success_rate(always_target, X, np.zeros(4, int), (6, 6, 1)))


class TestBackdoorLearner:
    def _train(self, attack_type, rounds=5):
        participants = make_synthetic_participants(n_clients=6, n_features=36, n_train=80, seed=3)
        learner = AttackedFederatedLearner(n_rounds=rounds, aggregation="fedavg", n_classes=2,
                                           attack_type=attack_type, byzantine_ids=[0, 1], seed=0,
                                           local_lr=0.1)
        return learner, participants, learner.train(participants, verbose=False)

    def test_success_rate_is_recorded_every_round(self):
        _, _, history = self._train("backdoor_trigger")
        assert all(0.0 <= h.attack_success_rate <= 1.0 for h in history)

    def test_other_attacks_do_not_record_it(self):
        _, _, history = self._train("label_flipping", rounds=2)
        assert all(h.attack_success_rate is None for h in history)

    def test_backdoor_raises_the_success_rate_over_a_clean_run(self):
        _, _, history = self._train("backdoor_trigger", rounds=10)
        learner, participants, _ = self._train("none", rounds=10)
        model = learner._make_model(36)
        X = np.concatenate([p.X_test for p in participants])
        y = np.concatenate([p.y_test for p in participants])
        clean = backdoor_success_rate(model.predict, X, y, (6, 6, 1))
        assert history[-1].attack_success_rate > clean + 0.2


class TestGridCells:
    def test_backdoor_is_outside_the_default_cells(self):
        from scripts.triage_grid import cells
        assert len(cells()) == 19
        assert all(t != "backdoor_trigger" for _, t in cells())
        assert cells(["backdoor_trigger"]) == [(a, "backdoor_trigger") for a in (0.5, 0.1, 0.05)]

    def test_triage_learner_records_the_success_rate(self):
        from src.triage import TriageLearner, TriageRule
        from src.triage.signals import SERVER_COSINE
        participants = make_synthetic_participants(n_clients=6, n_features=36, seed=3)
        learner = TriageLearner(n_rounds=2, n_classes=2, attack_type="backdoor_trigger", byzantine_ids=[0, 1],
                                seed=0, rule=TriageRule(signals=[SERVER_COSINE]))
        history = learner.train(participants, verbose=False)
        assert all(h.attack_success_rate is not None for h in history)
