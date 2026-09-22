import numpy as np
import pytest

from neuroplex.config import Config
from neuroplex.policy import MotorPolicy, MOTOR_CURRENTS


def test_motor_intent_comes_from_values_not_a_hidden_food_steering_rule():
    policy = MotorPolicy(Config(pretrained_policy=False, exploration_floor=0))
    senses = np.zeros(80)
    senses[14] = 0.8  # food to the right
    state = policy.encode(senses)
    # Deliberately prefer left. A hidden turn-toward-food reflex would ignore this.
    policy.values[state, 3] = 10
    np.testing.assert_array_equal(policy.begin(senses, False), MOTOR_CURRENTS[3])
    assert policy.action == 3
    policy.pending = False
    policy.values[state, 4] = 20
    np.testing.assert_array_equal(policy.begin(senses, False), MOTOR_CURRENTS[4])


def test_credit_goes_to_the_action_that_was_actually_taken():
    policy = MotorPolicy(Config(pretrained_policy=False))
    senses = np.zeros(80)
    policy.begin(senses, True)
    state, action = policy.state, policy.action
    for i in range(5):
        result = policy.observe(senses, int(i == 4), 0, True, True)
        if i < 4:
            assert result is None
            assert not policy.values.any()
    assert result > 0
    assert policy.values[state, action] > 0
    assert np.count_nonzero(policy.values) == 1
    assert policy.visits[state, action] == 1


def test_discounted_shaping_does_not_pay_for_repeating_a_visual_cycle():
    config = Config(pretrained_policy=False)
    policy = MotorPolicy(config)
    observations = []
    for brightness in (0.5, 0.9, 0.2, 0.5):
        senses = np.zeros(80)
        senses[8] = brightness
        observations.append(senses)
    gamma = config.policy_discount ** (config.world_dt / config.action_seconds)
    total = 0
    for i in range(3):
        policy.begin(observations[i], False)
        policy.observe(observations[i + 1], 0, 0, True, False)
        total += gamma**i * policy.last_shaping
    expected = gamma**3 * policy.potential(observations[-1]) - policy.potential(observations[0])
    assert total == pytest.approx(expected)
    assert total < 0


def test_terminal_state_has_no_bootstrap_or_potential_bonus():
    policy = MotorPolicy(Config(pretrained_policy=False))
    senses = np.zeros(80)
    senses[8] = 0.8
    policy.values.fill(100)
    policy.begin(senses, False)
    td = policy.observe(senses, 0, 0, False, True)
    assert policy.last_shaping == -policy.potential(senses)
    assert td == pytest.approx(policy.last_reward - 100)
    assert not policy.pending


def test_freezing_preserves_values_and_learning_counters():
    policy = MotorPolicy(Config())
    values, visits = policy.values.copy(), policy.visits.copy()
    updates = policy.updates
    assert updates == policy.bootstrap_updates == 2400
    for _ in range(20):
        senses = np.zeros(80)
        policy.begin(senses, False)
        policy.observe(senses, 1, 0, True, False)
    np.testing.assert_array_equal(policy.values, values)
    np.testing.assert_array_equal(policy.visits, visits)
    assert policy.updates == updates
    assert policy.exploration(False) == policy.config.exploration_floor
