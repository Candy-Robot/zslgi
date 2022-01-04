def test(rank, total_num_tests, gamma, tau, input_model, optimizer, env, log_dir):
    log = []
    torch.manual_seed(0)
    
    reward_sum = 0
    start_time = time.time()
    num_tests = 0
    reward_total_sum = 0
    state = env.reset()
    player = Agent(input_model, env, None, state) #chaosam fix this line

    player.state = player.env.reset()
    player.eps_len = 0
    
    flag = True
    max_score = 0
    while True:
        player.action_test()
        reward_sum += player.reward
        
        player.eps_len += 1

        if player.done:
            state = player.env.reset()
            
            num_tests += 1
            reward_total_sum += reward_sum
            reward_mean = reward_total_sum / num_tests
            
            log.append(
                "Time {0}, episode reward {1}, episode length {2}, reward mean {3:.4f}".
                format(
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(time.time() - start_time)),
                    reward_sum, player.eps_len, reward_mean))

            if reward_sum >= max_score: #save best score network
                max_score = reward_sum
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, '{0}{1}.dat'.format(
                        log_dir, "bestmodel"))

            reward_sum = 0
            player.eps_len = 0
            state = player.env.reset()
            player.state = state
            
            if(num_tests > total_num_tests):
              break

    print("Average performance over " + str(total_num_tests) + " episodes was " + str(reward_mean))

