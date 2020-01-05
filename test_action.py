import action as action_module


# Is none a valid to state?
def test_init():
    reward = 5
    action = action_module.Action(reward, None)
    assert reward == action.reward
    assert None is action.to
