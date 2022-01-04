#TEACHER NETWORK
import torch
from utils import *
import torch.nn as nn
import torch.nn.functional as F

shared_linear_numhidden = 800 #400
num_outputs = 13
newArchiTry = False #didn't work!

class teacherNetwork(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(teacherNetwork, self).__init__()
        if(not newArchiTry):
            #self.conv0 = nn.Conv2d(num_inputs, num_inputs*4, 3)
            self.conv1 = nn.Conv2d(num_inputs, 256, 2)
            self.conv2 = nn.Conv2d(256, 128, 1)

            self.shared_linear1 = nn.Linear(3200, 1600)
            self.shared_linear2 = nn.Linear(1600, shared_linear_numhidden)

            self.actor_linear = nn.Linear(shared_linear_numhidden, num_outputs)
            self.critic_linear = nn.Linear(shared_linear_numhidden, 1)
        else:
            self.conv1 = nn.Conv2d(num_inputs, 4, 1)
            self.conv2 = nn.Conv2d(4, 4, 1)
            
            maze_size_now = 4*grid_n*grid_n
            maze_size_half = 2*grid_n*grid_n
            just_grid = grid_n*grid_n
            
            self.shared_linear1 = nn.Linear(maze_size_now,maze_size_now)
            self.shared_linear2 = nn.Linear(maze_size_now,maze_size_half)
            self.shared_linear3 = nn.Linear(maze_size_half,just_grid)
            
            self.actor_linear = nn.Linear(just_grid, num_outputs)
            self.critic_linear = nn.Linear(just_grid, 1)
        
        self.apply(weights_init)

    def forward(self, inputs):
        x = inputs
        
        if(not newArchiTry):
            relu = True
            if(relu):
                #x = F.relu(self.conv0(x))
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = F.relu(self.shared_linear1(x))
                x = F.relu(self.shared_linear2(x))
            else:
                x = F.relu(self.conv1(inputs))
                x = F.relu(self.conv2(x))
                x = x.view(x.size(0), -1)
                x = F.sigmoid(self.shared_linear1(x))
                x = F.sigmoid(self.shared_linear2(x))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = F.relu(x)
            x = F.relu(self.shared_linear1(x))
            x = F.relu(self.shared_linear2(x))
            x = F.relu(self.shared_linear3(x))

        return self.actor_linear(x), self.critic_linear(x)