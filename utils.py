import numpy as np
import torch
import json
import logging
import os

instruction_list=["Transformed greenball",
  "Transformed tv",
  "Visited greenball",
  "Visited blueball",
  "Visited tv",
  "Visited heart",
  "Picked up greenball",
  "Picked up blueball",
  "Picked up tv",
  "Picked up heart"
]

imnames = ["empty",
"agent", #index 0 in our tensor
"greenball", #transformable0
#"greenguy", #transformable1
"blueball", #transformed0
#"yellowguy", #transformed1
"tv", #untransformable
"heart", #untransformable
"enemy",
"block",
"water"]

#change values depending on imnames
transformable_0_idx = 1
transformed_last_idx = 2
untransformable_0_idx = 3
untransformable_last_idx = 4

numTransformable = int((transformed_last_idx - transformable_0_idx + 1)/2)
numUntransformable = untransformable_last_idx - untransformable_0_idx + 1
enemy_idx = untransformable_last_idx + 1
block_idx = untransformable_last_idx + 2
water_idx = untransformable_last_idx + 3

# 创建环境
init_reset_total_count = 1
#all images have to be 32x32
blockSide = 32
#the grid is 10x10 blocks
grid_n = 6
numCellTypes = len(imnames)-1
episode_max_length = 300

numIterSoFar = 0
numIterToNoEntropyLoss = 3000
grid_depth = 8
act_dim = 13
numiters_between_reset_total_count_decrements = 500

act_types = ["Move ", "Pick up ", "Transform "]
act_dirs = ["N", "S", "W", "E"]

def actionNumToText(action):
  if action==12 or not isinstance(action, int):
    s = "None"
  elif(action < 0 or action > 12):
    s = "Invalid action"
  else:
    act_type = int(action/4) #0 move, 1 pickup, 2 transform
    act_dir = action - 4*act_type #0-3 is NSWE
    s = act_types[act_type] + act_dirs[act_dir]
  return s
  #1s are the verbs, 2s are the objects
subtask_1s = ["Visited ", "Picked up ", "Transformed "] #, "Picked up all", "Transformed all"] #will deal with "all" later
subtask_2s = imnames[transformable_0_idx+1:untransformable_last_idx+2]
subtask_1s_num = len(subtask_1s)
subtask_2s_num = len(subtask_2s)


#~~~~~~~~~~~~ CLEANUP ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def softremove(path):
  try:
    os.remove(path)
    return
  except Exception as e:
    return


def setup_logger(logger_name, log_file, level=logging.INFO):
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    l.setLevel(level)
    l.addHandler(fileHandler)
    l.addHandler(streamHandler)


def read_config(file_path):
    """Read JSON config."""
    json_object = json.load(open(file_path, 'r'))
    return json_object


def norm_col_init(weights, std=1.0):
    x = torch.randn(weights.size())
    x *= std / torch.sqrt((x**2).sum(1, keepdim=True))
    return x


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif not gpu:
            shared_param._grad = param.grad
        else:
            shared_param._grad = param.grad.cpu()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

