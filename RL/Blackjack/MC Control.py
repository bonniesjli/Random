def generate_episode_from_Q(env, Q, epsilon, nA):
    episode = []
    state = env.reset()
    while True:
        if state in Q:
            action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA))
        if state not in Q:
            action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if done:
            break
    return episode

#----------------------------------------------------------------------------------------------------
def get_probs(Q_s, epsilon, nA):
    """np.ones - creating an array given shape, filled with ones"""
    action_probability = np.ones(nA)
    action_probability = action_probability * epsilon / nA
    best_action = np.argmax(Q_s)
    action_probability[best_action] = 1 - epsilon + epsilon / nA
    return action_probability
#----------------------------------------------------------------------------------------------------
def update_Q(env, episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    discounts = np.array([gamma**i for i in range(len(rewards)+1)])
    # update the sum of the returns, number of visits, and action-value 
    # function estimates for each state-action pair in the episode
    for i, state in enumerate(states):
        old_q = Q[state][actions[i]]
        Q[state][actions[i]] = old_q + alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - old_q)
    return Q
#---------------------------------------------------------------------------------------------------------
def mc_control(env, num_episodes, alpha, gamma=1.0, eps_start=1.0, eps_decay=.99999, eps_min=0.05):
    nA = env.action_space.n
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(nA))
    epsilon = eps_start
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        epsilon = max(epsilon*eps_decay, eps_min)
        # generate an episode by following epsilon-greedy policy
        episode = generate_episode_from_Q(env, Q, epsilon, nA)
        # update the action-value function estimate using the episode
        Q = update_Q(env, episode, Q, alpha, gamma)
    # determine the policy corresponding to the final action-value function estimate
    policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return policy, Q 
