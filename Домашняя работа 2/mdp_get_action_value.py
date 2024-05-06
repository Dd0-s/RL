def get_action_value(mdp, state_values, state, action, gamma):
    """ Вычисляет Q(s,a) согласно формуле выше """
    
    q = 0
    for next_state, p in mdp.get_next_states(state, action).items():
        r = mdp.get_reward(state, action, next_state)
        q += p * (r + gamma * state_values[next_state])
    
    return q
