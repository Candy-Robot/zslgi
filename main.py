#MAIN TRAINING LOOP (main.py, modified)

import os
import argparse
from environment import MazeWorld
from utils import *
from model import teacherNetwork 
from train import train 
from test import test 
from shared_optim import SharedRMSprop, SharedAdam 
#from gym.configuration import undo_logger_setup
import time
import torch.multiprocessing as mp

os.environ["OMP_NUM_THREADS"] = "1"
# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# Implemented multiprocessing using locks but was not beneficial. Hogwild
# training was far superior

def get_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument('--savepath',default='models/test/')
  parser.add_argument('--instruction', type=int, default='9', help='instruction list')
  parser.add_argument('--num_iter', type=int, default=16001, help='num iter of every step')
  parser.add_argument('--wantToTest', default=False)
  parser.add_argument('--wantToTrain', default=True)
  parser.add_argument('--test_n', default=10)
  parser.add_argument('--loadpath',default='models/pick_heart/bestmodel_-1.pt')
  parser.add_argument('--gpu', default=True, help="use gpu")  
  parser.add_argument('--numWorkers', default=2, help="num workers")  
  opt = parser.parse_args()
  return opt
opt = get_opt()
instruction_opt = instruction_list[opt.instruction]

gpu = opt.gpu
wantToTest = opt.wantToTest
wantToTrain = opt.wantToTrain
load_path = opt.loadpath
numWorkers = opt.numWorkers

# 是否使用多线程
parallel = True
# 是否加载模型
loading = False

env = MazeWorld(grid_n,blockSide,numCellTypes, [instruction_opt], episode_max_length, \
                init_reset_total_count, gpu=gpu, changegrid=True)
env.reset()
env.render("step0", 0, "none", 0, "Start", 0)


# loadedModelPath = load_path

myLr = 0.0001
shared_model = teacherNetwork(grid_depth, act_dim)
# if(loading):
#     bestDict = torch.load(loadedModelPath)
#     shared_model.load_state_dict(bestDict)
shared_model.share_memory()
if(gpu):
    shared_model.cuda()
optimizer = SharedRMSprop(shared_model.parameters(), lr=myLr)


total_num_tests = 50
gamma = 0.99
tau = 0.96

loss_logs = []
rank  = 1
if(not gpu):
    rank = -1


torch.manual_seed(0)

# if(parallel and wantToTrain):
if(__name__  ==  '__main__'):

    if(gpu):
        torch.cuda.manual_seed(0)
        mp.set_start_method('spawn')#, force=True)
        print()
        
    processes = []

    #numWorkers = 8

    for rank in range(0, numWorkers):
        print(rank)
        if(not gpu):
            rank = -1
        p = mp.Process(
            target=train, args=(env, rank,  gamma, tau, shared_model, optimizer,5,10))#16001
        p.start()
        processes.append(p)
        time.sleep(0.1)

    for p in processes:
        time.sleep(0.1)
        p.join()
# elif(wantToTrain): #yes I know this is horrible
#     print("Start train " + instruction_opt)
#     # test(rank, total_num_tests, gamma, tau, shared_model, optimizer, env, 'sam')
#     # model, loss_log = train(env, rank,  gamma, tau, shared_model, optimizer,num_iter = opt.num_iter, setHardIter=7000, num_steps=1000)
#     torch.autograd.set_detect_anomaly(True)
#     model, loss_log = train(env, rank,  gamma, tau, shared_model, optimizer,num_iter = 5, setHardIter=10, num_steps=10)
#     loss_logs = loss_logs + loss_log
#     # test(rank, total_num_tests, gamma, tau, shared_model, optimizer, env, 'sam')


# if(wantToTrain and __name__  ==  '__main__'):
    #samget
    log_dir = opt.savepath
    state_to_save = shared_model.state_dict()
    torch.save(state_to_save, '{0}{1}.pt'.format(log_dir, "finalModel"))
    print("over")