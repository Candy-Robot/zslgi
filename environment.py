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

#workaround to avoid register_extensions PIL error
from PIL import Image
def register_extension(id, extension): Image.EXTENSION[extension.lower()] = id.upper()
Image.register_extension = register_extension
def register_extensions(id, extensions): 
  for extension in extensions: register_extension(id, extension)
Image.register_extensions = register_extensions

mode = "zslgi"

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import torch
import os,sys
import datetime
import imageio
from pprint import pprint
import time
import datetime

####################################################################
########################## SETUP ###################################
####################################################################
#~~~~~~~~~~~ FUNDAMENTAL VARIABLES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#all images have to be 32x32
blockSide = 32
#the grid is 10x10 blocks
grid_n = 6

#~~~~~~~~~~~ IMAGES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tilesDir = "content/"

#there should be 18 elements + 1 "empty" cell
numCellTypes = len(imnames)-1
grid_depth = numCellTypes
assert(grid_depth==8)
episode_max_length = 300

act_dim = 13

#~~~~~~~~~~~~ Mapping b/t actions, subtasks and integers
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

for fname in os.listdir('./'):
    if fname.startswith(mode):
        softremove(os.path.join('.', fname))

####################################################################
################## IMPORT IMAGES ###################################
####################################################################
#print(imnames.index("hat")) #to get the reverse mapping objectname->index

assert(len(imnames)==numCellTypes+1)

imlist = []
for i in range(numCellTypes+1):
  imlist.append(Image.open(tilesDir + imnames[i] + ".jpg"))
emptyim = imlist[0]
agent = imlist[1]

greyimg = Image.open(tilesDir + "grey.jpg")

####################################################################
################## MAZEWORLD CLASS #################################
####################################################################
class MazeWorld(object):
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #initialize the world~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def __init__(self, grid_n, blockSide, numCellTypes, instruction, maxnumsteps, reset_total_count, gpu=False, useEnemy=False, changegrid=False):
    self.grid_n = grid_n
    self.blockSide = blockSide
    self.numCellTypes = numCellTypes
    self.instruction = instruction
    self.originalinstruction = instruction
    assert(type(instruction)==type([]))
    self.maxnumsteps = maxnumsteps
    self.numsteps = 0
    self.changegrid = changegrid
    self.gpu = gpu
    self.done = False

    #initialize the grid
    self.initgrid = self.buildgrid(grid_n, instruction)

    #place agent
    #CHAO EDIT THIS PLS
    while True:
      i = int(grid_n * np.random.uniform())
      j = int(grid_n * np.random.uniform())
      if(self.initgrid[i][j] < block_idx+1):
        break
    self.agent_i = i
    self.agent_j = j
    self.initagent_i = i
    self.initagent_j = j
    #print("initially i and j are " + str(i) + "and" + str(j))

    #make tensor
    self.t = self.initgridToTensor()
    if(gpu):
        self.t = self.t.cuda()

    #variables for enemy
    self.useEnemy = useEnemy
    self.enemy_lifelength = 10 #how many steps it stays
    self.enemy_prob = .05 #prob at each step of appearing
    self.enemy_age = -1
    self.enemy_i = -1
    self.enemy_j = -1
    
    #keep track of how many times we have reset
    self.reset_total_count = reset_total_count
    self.reset_current_count = 0
    
  def reset(self, instructions=None, easy = False, hard = False):
    if(instructions!=None):
      self.instruction = instructions
      self.originalinstruction = instructions
    else:
      self.instruction = self.originalinstruction
    self.numsteps = 0
    
    assert(not (easy and hard))
    
    
    for ins in self.instruction:
      lastword = ins.split()[-1]
      objIndex = imnames.index(lastword)
        
    if(self.changegrid or instructions!=None):
      #CHAO EDIT THIS PLS
      if(self.reset_current_count >= self.reset_total_count):

          #initialize the grid
          self.initgrid = self.buildgrid(grid_n, self.originalinstruction)
          self.initgrid[self.agent_i][self.agent_j] = 0

          #place agent
          for itr in range(1000):
            i = int(grid_n * np.random.uniform())
            j = int(grid_n * np.random.uniform())
            min_dist = float("Inf")
            max_dist = 0
            if(self.initgrid[i][j] < block_idx+1):
                ### add constrants for the env
                if easy: # 目标的距离距离agent只有3步的距离
                    for col in range(grid_n):
                        for row in range(grid_n):
                            if self.initgrid[col][row] == objIndex:
                                temp_dist = (np.abs(col-i) + np.abs(row-j))
                                if temp_dist > max_dist:
                                    max_dist = temp_dist

                    if max_dist <= 3:
                        break
                    
                elif hard: # 目标的距离距离agent大于3步的距离
                    for col in range(grid_n):
                        for row in range(grid_n):
                            if self.initgrid[col][row] == objIndex:
                                temp_dist = (np.abs(col-i) + np.abs(row-j))
                                if temp_dist < min_dist:
                                    min_dist = temp_dist

                    if min_dist > 3:
                        break
                else:
                    break
                        
            if itr == 999:
                if(easy):
                    print ("Resetting with easy constraint not working after 1k tries, trying again")
                elif(hard):
                    print ("Resetting with hard constraint not working after 1k tries, trying again")
                else:
                    print("Resetting with no constraint not working after 1k tries, trying again")
                self.reset()

          self.agent_i = i
          self.agent_j = j
            
          self.reset_current_count = 0
    else:
      self.agent_i = self.initagent_i
      self.agent_j = self.initagent_j
      #print("Resetting; now i and j are " + str(self.agent_i) + "and" + str(self.agent_j))

    #make tensor
    self.t = self.initgridToTensor()
    if(self.gpu):
        self.t = self.t.cuda()

    #variables for enemy
    #self.useEnemy stays the same
    self.enemy_age = -1
    self.enemy_i = -1
    self.enemy_j = -1
    
    self.done = False
    
    self.reset_current_count = self.reset_current_count + 1
    
    return self.t
    
  #sambuildgrid
  def buildgrid(self, grid_n, instruction):
    #some values to support random env generation (assumes 18 objs)
    prob_nonempty = .3 #prob that a cell is not empty
    prob_water_ifnonempty = .25
    prob_block_ifnonemptynonwater = .45
    prob_untransformable_ifnothingelse = numUntransformable/(numUntransformable+numTransformable)

    grid = np.zeros((grid_n,grid_n), dtype=int)

    #make sure objects in instructions exist in the grid!
    for ins in instruction:
      lastword = ins.split()[-1]
      objIndex = imnames.index(lastword)

      while True:
        i = int(grid_n * np.random.uniform())
        j = int(grid_n * np.random.uniform())
        if(grid[i][j] == 0):
          grid[i][j] = objIndex
          break

    for i in range(grid_n):
      for j in range(grid_n):
        if(np.random.uniform()<prob_nonempty and grid[i][j]==0):
          #square is nonempty
          if(np.random.uniform()<prob_water_ifnonempty):
            grid[i][j] = water_idx+1 # water
          else:
            if(np.random.uniform()<prob_block_ifnonemptynonwater):
              grid[i][j] = block_idx+1 #block
            else:
              if(np.random.uniform()<prob_untransformable_ifnothingelse):
                grid[i][j] = 1+untransformable_0_idx + int(numTransformable*np.random.uniform())
                #nontransformable
              else:
                grid[i][j] = 1+transformable_0_idx+int(numTransformable*np.random.uniform())
                #o/w pick transformable

    return grid

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #act in the world~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def step(self, action):
    if(self.done):
        raise ValueError('Trying to step in a done env.')
    
    reward = -.1
    done = False
    taskupdate = ""

    #execute action
    if(action != 12): #12 is no op
      act_type = int(action/4) #0 move, 1 pickup, 2 transform
      act_dir = action - 4*act_type #0-3 is NSWE

      #get the cell the action is "acting on"
      act_i = self.agent_i
      act_j = self.agent_j
      if(act_dir==0):
        act_i -= 1
      elif(act_dir==1):
        act_i += 1
      elif(act_dir==2):
        act_j -= 1
      else: #(act_dir==1)
        act_j += 1

      #check that the "acted on" cell is not OOB or a block
      if(not(act_j <0 or act_i <0         or act_j>=self.grid_n or act_i>=self.grid_n)):
        if(self.t[block_idx][act_i][act_j] != 1):
          #deal with action
          if(act_type==0): #move
            self.t[0][act_i][act_j] = 1
            self.t[0][self.agent_i][self.agent_j] = 0
            self.agent_i = act_i
            self.agent_j = act_j

            #if we visited an object, change taskupdate
              #start by finding out what object is there
            objidx = -1
            for idx in range(1, self.numCellTypes-1):
              if(self.t[idx][act_i][act_j]==1):
                objidx = idx
                break
            if(objidx!=-1):
              taskupdate = "Visited " + imnames[objidx+1]
          else: #pick up or transform
            if(act_type==1):
              action_name = "Picked up "
            elif(act_type==2):
              action_name = "Transformed "
            
            #start by finding out what object is there
            objidx = -1
            for idx in range(1, self.numCellTypes):
              if(self.t[idx][act_i][act_j]==1):
                objidx = idx
                break

            #we know object and where it is, let's execute
            if(objidx != -1 and objidx != water_idx):
              didSomething = False

              #if transformable and transform action is taken,
              #we need to "replace" object with its transformation
              if(objidx >= transformable_0_idx and objidx < transformable_0_idx+numTransformable):
                if(act_type==2):
                  self.t[objidx+numTransformable][act_i][act_j]=1
                  self.t[objidx][act_i][act_j]=0 #remove object
                  didSomething = True

              #LATER ADDITION: DECIDED TO MAKE TV TRANSFORM TO HEART
              if(objidx ==  untransformable_last_idx-1 and act_type==2):
                  self.t[untransformable_last_idx][act_i][act_j]=1
                  self.t[objidx][act_i][act_j]=0 #remove object
                  didSomething = True
                
              if(act_type==1):
                self.t[objidx][act_i][act_j]=0 #remove object                
                didSomething = True

              if(objidx==enemy_idx): #enemy
                self.enemy_age = 1 #so we can clean up
                if(act_type==2): #transform
                  reward += .9
                didSomething = True

              if(didSomething):
                taskupdate = action_name + imnames[objidx+1]


    #deal with enemy
    if(self.useEnemy):
      if(self.enemy_age==-1):
        if(self.enemy_prob > np.random.uniform()):
          #create enemy
          countloop = 0
          while True:
            self.enemy_i = int(grid_n * np.random.uniform())
            self.enemy_j = int(grid_n * np.random.uniform())
            fibre = self.t[:,self.enemy_i,self.enemy_j]
            if(int(sum(fibre)) == 0):
              self.t[enemy_idx][self.enemy_i][self.enemy_j] = 1
              self.enemy_age = self.enemy_lifelength
              break
            countloop += 1
            if(countloop==100):
              print("Looping too many times for enemy creation!")
              break
      else:
        self.enemy_age -= 1
        if(self.enemy_age==0): #enemy's life has ended
          self.enemy_age = -1
          self.t[enemy_idx][self.enemy_i][self.enemy_j] = 0
          self.enemy_i = -1
          self.enemy_j = -1
        else: #move the enemy randomly
          #generate the 4 possible cells the agent could move to
          dirs = [[self.enemy_i, self.enemy_j] for x in range(4)]
          dirs[0][1] += 1
          dirs[1][1] -= 1
          dirs[2][0] += 1
          dirs[3][0] -= 1

          legaldirs = []

          for d in dirs:
            if(d[0]>=0 and d[0] < self.grid_n and             d[1]>=0 and d[1] < self.grid_n):
              fibre = self.t[:,d[0],d[1]]
              if(int(sum(fibre)) == 0):
                legaldirs.append(d)

          if(len(legaldirs)!=0):
            self.t[enemy_idx][self.enemy_i][self.enemy_j] = 0
            newdir = random.choice(legaldirs)
            self.enemy_i = newdir[0]
            self.enemy_j = newdir[1]
            self.t[enemy_idx][self.enemy_i][self.enemy_j] = 1

    #if in water, punish!
    if(self.t[water_idx][self.agent_i][self.agent_j] == 1):
      reward -= .3

    #reward + 1 and terminate if instruction is done
    if(self.instruction[0] == taskupdate):
      self.instruction = self.instruction[1:]
      if(len(self.instruction)==0):
        done = True
        self.done = True
        reward += 0.9

    #increment numsteps and terminate if numsteps > maxnumsteps
    self.numsteps += 1
    if(self.numsteps > self.maxnumsteps):
      done = True

    return self.t, reward, done, taskupdate

  #load a tensor, throws an error if exists enemy
  def loadTensor(self, newT, newInstruction):
    #set tensor
    self.t = newT
    #set agent position
    foundAgent = False
    for i in range(self.grid_n):
      for j in range(self.grid_n):
        if(self.t[0][i][j] == 1):
          foundAgent = True
          self.agent_i = i
          self.agent_j = j
          
    assert(foundAgent) #if no agent in grid something is wrong
    
    #throw error if there is an enemy
    for i in range(self.grid_n):
      for j in range(self.grid_n):
        if(self.t[enemy_idx][i][j] == 1):
          raise ValueError('Code cannot handle loading grid with agent yet.')
    self.useEnemy = False
    
    #initgrid is none because load could be from any initgrid
    self.initgrid = False
    
    #reload instruction, set initinstruction to none for same reason as above
    self.originalinstruction = None
    self.instruction = newInstruction
    
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #render the world~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def makeImageFromGrid(self, gridarr, step, action, reward, taskupdate, cumreward):
    assert(self.grid_n == gridarr.shape[1])

    result_width = int(self.blockSide*self.grid_n*1.5)
    result_height = self.blockSide*(self.grid_n+1)
    result = Image.new('RGB', (result_width, result_height))

    #DRAW GRID
    for i in range(self.grid_n):
      for j in range(self.grid_n):
        itemindex = gridarr[j][i]
        result.paste(im=imlist[itemindex], box=(i*self.blockSide, j*self.blockSide))

    #DRAW GREY AREA FOR INSTRUCTIONS
    for i in range(self.grid_n, int(self.grid_n*1.5)):
      for j in range(self.grid_n):
        result.paste(im=greyimg, box=(i*self.blockSide, j*self.blockSide))

    #WRITING AND LINES
    #font_type = ImageFont.truetype("arial.ttf", 12, encoding="unic")
    draw = ImageDraw.Draw(result)
      #bottom line
    draw.line(xy=[0, self.blockSide*self.grid_n, self.blockSide*self.grid_n,      self.blockSide*self.grid_n], fill=(255,255,255))
      #bottom text
    mytext1 = "Step: " + str(step) + ", act: " + actionNumToText(action) + ", rew: " +       str(reward) + ", cumrew = " + str(cumreward+reward)
    mytext2 = "Taskupdate: " + taskupdate
    draw.text(xy =(0, self.blockSide*(self.grid_n)),       text=mytext1, fill=(255,255,255))
    #, font=font_type)
    draw.text(xy =(0, self.blockSide*(self.grid_n+.5)),       text=mytext2, fill=(255,255,255))
              #, font=font_type)
      #instruction text
    num_instructions = len(self.originalinstruction)
    current_instruction = num_instructions - len(self.instruction)
    for i in range(num_instructions):
      if(i!=current_instruction):
        draw.text(xy = (self.grid_n*self.blockSide, i*self.blockSide/3),           text=self.originalinstruction[i], fill=(0,0,0))#, font = font_type)
      else:
        draw.text(xy = (self.grid_n*self.blockSide, i*self.blockSide/3),           text=self.originalinstruction[i], fill=255)#, font = font_type)

    return result

  def makeTextImage(self, mytext):
    result_width = int(self.blockSide*self.grid_n*1.5)
    result_height = self.blockSide*(self.grid_n+1)
    result = Image.new('RGB', (result_width, result_height))
    #font_type = ImageFont.truetype("arial.ttf", 32, encoding="unic")
    draw = ImageDraw.Draw(result)
    draw.text(xy =(result_width*.4, result_height*.4),       text=mytext, fill=(255,255,255))#, font=font_type)
    return result

  def render(self, filename, step, action, reward, taskupdate, cumreward):
    #place agent
    newgrid, self.agent_i, self.agent_j = self.tensorToArr()
    newgrid[self.agent_i][self.agent_j] = 1

    result = self.makeImageFromGrid(newgrid, step, action, reward,     taskupdate, cumreward)
    outfile = filename + ".jpg"
    #result.show()
    result.save(outfile)

  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #switch b/t representations of the world~~~~~~~~~~~~~~~~~~~~~~~~
  #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  def initgridToTensor(self):
    t = torch.zeros((self.numCellTypes,self.grid_n,self.grid_n))#, dtype=torch.int32)

    for i in range(self.grid_n):
      for j in range(self.grid_n):
        idx = self.initgrid[i][j]
        if(idx!=0):
          t[idx-1][i][j] = 1

    t[0][self.agent_i][self.agent_j] = 1 #place agent

    return t

  def tensorToArr(self):
    arr = np.zeros((self.grid_n,self.grid_n), dtype=int)

    for ct in range(self.numCellTypes):
      for i in range(self.grid_n):
        for j in range(self.grid_n):
          if(self.t[ct][i][j] == 1):
            if(ct == 0):
              agent_i = i
              agent_j = j
            else:
              arr[i][j] = ct+1
    return arr, agent_i, agent_j