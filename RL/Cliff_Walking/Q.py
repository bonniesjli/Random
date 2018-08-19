def q_learning(env, num_episodes, alpha, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        # begin an episode, observe S
        state = env.reset()
        #get action probabilities
        action_probabilities = epsilon_greedy_probs(env, Q[state], i_episode, eps=None)
        #pick action A
        action = np.random.choice(np.arange(env.nA), p = action_probabilities)
        #limit number of time step per episode
        for t_step in np.arange(300):
            next_state, reward, done, info = env.step(action)
            if not done:
                # get epsilon-greedy action probabilities
                action_probabilities = epsilon_greedy_probs(env, Q[next_state], i_episode, eps=None)
                # pick next action A'
                next_action = np.random.choice(np.arange(env.nA), p = action_probabilities)
                # update TD estimate of Q
                ## Q[state][action], Q[next_state][next_action], reward, alpha, gamma
                Q[state][action] = Q[state][action] + alpha *(reward + gamma* np.max(Q[next_state])
                                                                - Q[state][action])
                # S <- S'
                state = next_state
                # A <- A'
                action = next_action
            if done:
                # update TD estimate of Q
                Q[state][action] = Q[state][action] + alpha *(reward - Q[state][action])        
    return Q
