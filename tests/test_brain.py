import numpy as np

from neuroplex.brain import Brain
from neuroplex.config import Config


def test_reward_sign_and_eligibility_control_weight_updates():
    brain = Brain(Config())
    synapse = np.flatnonzero(brain.plastic)[0]
    original = brain.weights.copy()
    brain.eligibility.fill(0)
    brain.eligibility[synapse] = 1
    brain.reinforce(0.2)
    assert brain.weights[synapse] > original[synapse]
    assert np.count_nonzero(brain.weights != original) == 1
    brain.weights[:] = original
    brain.reinforce(-0.2)
    assert brain.weights[synapse] < original[synapse]


def test_causal_spike_pair_leaves_positive_trace_and_reverse_is_negative():
    brain = Brain(Config())
    k = np.flatnonzero(brain.plastic & (brain.pre < 32))[0]
    pre, post = brain.pre[k], brain.post[k]
    brain.v.fill(-1)
    brain.v[post] = 2
    brain.pre_trace[pre] = 1
    brain.step(np.zeros(80))
    assert brain.spikes[post]
    assert brain.eligibility[k] > 0
    brain.reset_activity()
    brain.v.fill(-1)
    brain.v[pre] = 2
    brain.post_trace[post] = 1
    brain.step(np.zeros(80))
    assert brain.spikes[pre]
    assert brain.eligibility[k] < 0


def test_frozen_synapses_and_inhibitory_sign_survive_activity_and_rewards():
    brain = Brain(Config())
    assert len(brain.weights) == 16000
    assert np.all(brain.pre != brain.post)
    assert brain.inhibitory.sum() == 64
    brain.learning = False
    original = brain.weights.copy()
    for _ in range(15):
        brain.advance(np.ones(80))
        brain.reinforce(0.5)
    np.testing.assert_array_equal(original, brain.weights)
    brain.learning = True
    for reward in [100, -100]:
        brain.reinforce(reward)
        assert np.all(brain.weights[brain.plastic] >= 0)
        assert np.all(brain.weights[brain.plastic] <= brain.config.weight_max)
        np.testing.assert_array_equal(original[~brain.plastic], brain.weights[~brain.plastic])
    assert np.isfinite(brain.v).all()
