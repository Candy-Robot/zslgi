#TRAIN FUNCTION!
#from setproctitle import setproctitle as ptitle
import torch
import torch.optim as optim
from utils import ensure_shared_grads
from model import teacherNetwork
from player_util import Agent
from torch.autograd import Variable
import time
import datetime
from utils import *

numIterSoFar = 0
numIterToNoEntropyLoss = 3000
grid_depth = 8
act_dim = 13
numiters_between_reset_total_count_decrements = 500

#def train(rank, args, shared_model, optimizer, env_conf):
# def train(env, rank, gamma, tau, model, optimizer, num_iter, setHardIter, num_steps=1000):
def train(env, rank, gamma, tau, model, optimizer, num_iter, setHardIter, num_steps=5):
    global numIterSoFar
    global numIterToNoEntropyLoss
    # log_dir = opt.savepath
    log_dir = 'models/test/'

    if(rank>-1):
        torch.cuda.manual_seed(rank)
    else:
        torch.manual_seed(rank)

    lossLog = []

    state = env.reset() # 根据任务生成地图
    player = Agent(None, env, None, state) # 初始化智能体，输入地图和环境
    player.model = teacherNetwork(grid_depth, act_dim)

    if(rank>-1):
        player.model.cuda()
        # player.model

    numDone = 0
    totalDone = 0
    ep_lens = []
    print("Process " + str(rank) + " starting training" + ", time is ", end=" ")
    print(datetime.datetime.now().strftime('%H-%M-%S'))
    
    num_eps_per_iter = 1
    bestAvgLen = 1000
    bestAvgLen_score = 0
    
    hardSetting = False

    for iter in range(num_iter):
        #if(iter % setHardIter == 0 and iter > 0):
        #    hardSetting = True #make things hard after the 6kth iteration
        #    print("Making it hard from now on!!!!")
        
        annealing = max(0, 1-numIterSoFar/numIterToNoEntropyLoss)
        
        if(iter%numiters_between_reset_total_count_decrements==0 and iter > 0 and env.reset_total_count>1):
            #env.decrement_reset_total_count() if I had time to build it
            env.reset_total_count -= 1

        if(iter%100==0 and iter > 0):
          if(len(ep_lens)>0):
            avgLen = sum(ep_lens)/len(ep_lens)
            if(bestAvgLen>avgLen):
                bestAvgLen = avgLen
                bestAvgLen_iter = int(iter)
                bestAvgLen_score = str(numDone) + " out of " + str(totalDone)
                state_to_save = player.model.state_dict()
                torch.save(state_to_save, '{0}{1}{2}.pt'.format(
                        log_dir, "bestmodel_", rank))
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Process " + str(rank) + " Training; current iter: " + str(iter) + ", time is ", end=" ")
            print(datetime.datetime.now().strftime('%H-%M-%S'))
            print("Number of times it's done before maxsteps is " + str(numDone) + " out of " + str(totalDone))
            print("Average episode length is " + str(avgLen))
            print("So far bestAvgLen is " + str(bestAvgLen) + " achieved at iter " + \
                  str(bestAvgLen_iter) + " with " + bestAvgLen_score)
            ep_lens = []
            numDone = 0
            totalDone = 0

        player.model.load_state_dict(model.state_dict())
        
        total_loss = 0
        
        for x in range(num_eps_per_iter):
            for step in range(num_steps):
              player.action_train()
              player.eps_len = player.eps_len + 1
              if player.done:
                totalDone = totalDone + 1
                ep_lens.append(player.eps_len)
                player.eps_len = 0

                if(player.rewards[-1]>0):
                  numDone = numDone + 1
                state = player.env.reset(hard = hardSetting)
                player.state = state
                player.done = False #after reset

                break            

            if(rank>-1):
                player.values.append(torch.zeros(1, 1).cuda())
            else:
                player.values.append(torch.zeros(1, 1))

            policy_loss = 0
            value_loss = 0

            if(rank>-1):
                gae = torch.zeros(1, 1).cuda()
            else:
                gae = torch.zeros(1, 1)

            #GAE Calc
            for i in reversed(range(len(player.rewards))):
                R = player.rewards[i]
                advantage = R - player.values[i]
                value_loss = value_loss + advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = player.rewards[i] + gamma *                 player.values[i + 1].data - player.values[i].data

                gae = gae * gamma * tau + delta_t

                policy_loss = policy_loss -                 player.log_probs[i] *                 Variable(gae) - annealing * .05 * player.entropies[i]

            #Calculate loss and backprop
            total_loss = total_loss +  policy_loss + 0.5 * value_loss
            lossLog.append(float(policy_loss + 0.5 * value_loss))
            player.clear_actions()

        player.model.zero_grad()
        (total_loss).backward()
        ensure_shared_grads(player.model, model)
        optimizer.step()
        
        numIterSoFar = numIterSoFar + 1
    
    #once it's done, save the final network
    state_to_save = player.model.state_dict()
    torch.save(state_to_save, '{0}{1}{2}.pt'.format(
        log_dir, "finalmodel_", rank))
    
    return player.model, lossLog