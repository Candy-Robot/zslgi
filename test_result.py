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
import torch.nn.functional as F
import imageio

mode = "zslgi"
def get_opt():
  parser = argparse.ArgumentParser()
  parser.add_argument('--savepath',default='models/test/')
  parser.add_argument('--instruction', type=int, default='9', help='instruction list')
  parser.add_argument('--num_iter', type=int, default=16001, help='num iter of every step')
  parser.add_argument('--wantToTest', default=False)
  parser.add_argument('--wantToTrain', default=True)
  parser.add_argument('--test_n', default=5)
  parser.add_argument('--loadpath',default='models/test/bestmodel_1.pt')
  parser.add_argument('--gpu', default=True, help="use gpu")  
  parser.add_argument('--numWorkers', default=2, help="num workers")  
  opt = parser.parse_args()
  return opt
opt = get_opt()

load_path = opt.loadpath
instruction_opt = instruction_list[opt.instruction]
gpu = opt.gpu

env = MazeWorld(grid_n,blockSide,numCellTypes, [instruction_opt], episode_max_length, \
                    init_reset_total_count, gpu=gpu, changegrid=True)

env.reset()
env.render("step0", 0, "none", 0, "Start", 0)
loading = True
loadedModelPath = load_path
if(__name__  ==  '__main__'):
    #print(__name__)
    #print("doing this")
    myLr = 0.0001
    shared_model = teacherNetwork(grid_depth, act_dim)
    if(loading):
        bestDict = torch.load(loadedModelPath)
        shared_model.load_state_dict(bestDict)
    shared_model.share_memory()
    if(gpu):
        shared_model.cuda()
    optimizer = SharedRMSprop(shared_model.parameters(), lr=myLr)

#RUN n EPISODEs AND SAVE IT WITH NETWORK
#samback
rank = 1

if(__name__  ==  '__main__' ):
    #print("hello, world")
    run_n_episodes = True
    n = opt.test_n
    avgNumSteps = []
    printingresults = False
    makeimages = False #make an image for each step
    justprintsummary = True #only print numsteps and cumreward @ the end
    makegif = True

    if(run_n_episodes):
        numSolved = 0
        for x in range(n):
            
            state = env.reset()
            submode = mode+"_" + str(x) + "_" + str(rank) + "_"

            if(makeimages or makegif):
              for fname in os.listdir('./'):
                if fname.startswith(submode) and fname.endswith(".jpg"):
                    softremove(os.path.join('.', fname))
              env.render(submode+ "0", 0, "none", 0, "Start", 0)
            if(logging):
              envlog = torch.zeros((episode_max_length+3,env.numCellTypes,env.grid_n,env.grid_n))#,\
               #dtype=torch.int32)
              envlog[0] = env.t
              actRewDoneLog = []


            model = shared_model
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #                          Run an episode
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            done = False
            i = 1
            cumreward = 0

            while not done:
              logit, value = model(state.unsqueeze(0))
              prob = F.softmax(logit, dim=1)
              action = prob.multinomial(1).data 

              state, reward, done, taskupdate = env.step(int(action[0]))
              renderfilename = submode + str(i)

              if(logging):
                envlog[i] = env.t
                actRewDoneLog.append([action, reward, done])

              actText = actionNumToText(int(action[0]))

              if(printingresults):
                print(str(i) + ": Action was " + actText +       ", reward is " + str(reward) + ", taskupdate is "       + taskupdate + ".")

              if(makeimages or makegif):
                env.render(renderfilename, i, int(action[0]), reward, taskupdate, cumreward)

              i += 1
              cumreward += reward

            if(reward>0):
                numSolved += 1
            else:
                print("Episode " + str(x) + "didn't solve. See " + submode + ".gif for episode") 
                
            if(printingresults or justprintsummary):
              print("Ep " + str(x) + ": Env terminated after " + str(i-1) + " steps, cumreward = " + str(cumreward))
            
            avgNumSteps.append(i-1)

            if(logging):
              softremove('envlog.tensor')
              torch.save(envlog, 'envlog.tensor')

            if(makegif):
              #code from Alex Buzunov at https://goo.gl/g2G9c6
              duration = 1
              filenames = sorted(filter(os.path.isfile, [x for x in os.listdir()   if (x.endswith(".jpg") and x.startswith(submode))]), key=lambda   p: os.path.exists(p) and os.stat(p).st_mtime or   time.mktime(datetime.now().timetuple()))

              #make an image that says "START"
              result = env.makeTextImage("START")
              result.save(submode + "Start.jpg")

              #make an image that says "END"
              result = env.makeTextImage("END")
              result.save(submode + "End.jpg")
              filenames.append(submode+"End.jpg")

              images = []
              images.append(imageio.imread(submode + "Start.jpg"))
              for filename in filenames:
                images.append(imageio.imread(filename))

              #output_file = 'Gif-%s.gif' % datetime.datetime.now().strftime('%Y-%M-%d-%H-%M-%S')
              output_file = submode + '.gif'
              imageio.mimsave(output_file, images, duration=duration)

            
            #get rid of all the jpgs that start with submode
            if(makeimages or makegif):
              for fname in os.listdir('./'):
                if fname.startswith(submode) and fname.endswith(".jpg"):
                    softremove(os.path.join('.', fname))
        
        print("numSolved is " + str(numSolved) + " out of " + str(n))
        print("Average number of steps is " + str(sum(avgNumSteps)/len(avgNumSteps)))