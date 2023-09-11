import random
import numpy as np
import pandas as pd
# from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
# import torch.optim as optim
# import copy
from ursina import distance
DEVICE = 'cpu'


class SwarmTargetDQNAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.reward = 0
        self.gamma = 0.7
        self.choice = 0
        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])
        self.agent_target = 1
        self.agent_predict = 0
        self.learning_rate = params['learning_rate']
        self.epsilon = 1
        self.actual = []
        self.first_layer = params['first_layer_size']
        self.second_layer = params['second_layer_size']
        self.third_layer = params['third_layer_size']
        self.memory = collections.deque(maxlen=params['memory_size'])
        self.weights = params['weights_path']
        self.load_weights = params['load_weights']
        self.optimizer = None
        self.network()

    def network(self):
        # Layers
        self.f1 = nn.Linear(9, self.first_layer)
        self.f2 = nn.Linear(self.first_layer, self.second_layer)
        self.f3 = nn.Linear(self.second_layer, self.third_layer)
        self.f4 = nn.Linear(self.third_layer, 3)

        if self.load_weights:
            self.model = self.load_state_dict(torch.load(self.weights))
            print("weights loaded")

    def forward(self, x):
        x = F.leaky_relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.softmax(self.f4(x))
        return x

    def get_state(self, unit, targets, walls, swarm_count, target_count, base_scale):
        """
        - Plan, reduce the vision down to a few neurons of "Unit seen, wall seen, or target seen, left, right, center"
        Return the state.
        The state is a numpy array of 9 values, representing:
            - 0, 1: Rays left see entity => target or wall
            - 2, 3: Center ray sees entity => target or wall
            - 4, 5: Rays right see entity => target or wall
            - target proximity detection
            - wall proximity detection
            - target hit detection
            - wall hit detection
        """

        """
                if unit.vision[0].entity is not None:
                    a = unit.vision[0].entity
                else:
                    a = False

                if unit.vision[1].entity is not None:
                    b = unit.vision[1].entity
                else:
                    b = False

                if unit.vision[2].entity is not None:
                    c = unit.vision[2].entity
                else:
                    c = False

                if unit.vision[3].entity is not None:
                    d = unit.vision[3].entity
                else:
                    d = False

                if unit.vision[4].entity is not None:
                    e = unit.vision[4].entity
                else:
                    e = False

                if unit.vision[5].entity is not None:
                    f = unit.vision[5].entity
                else:
                    f = False

                if unit.vision[6].entity is not None:
                    g = unit.vision[6].entity
                else:
                    g = False
                    
                    
                    
                    (a and a.name),  # Ray 0 identifies entity

            (a and float(distance(unit, a))),  # Ray 0 identifies entity distance from the unit

            (a and a.color),   # Ray 0 identifies entity colour

            (b and b.name),  # Ray 1

            (b and float(distance(unit, b))),  #''

            (b and b.color),   #''

            (c and c.name),  # Ray 2

            (c and float(distance(unit, c))),  #''

            (c and c.color),   #''

            (d and d.name),  # Ray 3

            (d and float(distance(unit, d))),  #''

            (d and d.color),   #''

            (e and e.name),  # Ray 4

            (e and float(distance(unit, e))),  #''

            (e and e.color),   #''

            (f and f.name),  # Ray 5

            (f and float(distance(unit, f))),  #''

            (f and f.color),   #''

            (g and g.name),  # Ray 6

            (g and float(distance(unit, g))),  #''

            (g and g.color),   #''
            
            match state[i]:
                case 'target':
                    state[i] = 2
                case 'unit':
                    state[i] = 3
                case 'North':
                    state[i] = 4
                case 'South':
                    state[i] = 5
                case 'East':
                    state[i] = 6
                case 'West':
                    state[i] = 7
                case 'red':
                    state[i] = 8
                case 'white':
                    state[i] = 9
                case 'blue':
                    state[i] = 10
                case'white10':
                    state[i] = 11
                case 'wall':
                    state[i] = 12
                case _:
        """

        vision_range = len(unit.vision)
        left_target = False
        left_wall = False
        right_target = False
        right_wall = False
        center_target = False
        center_wall = False
        target_hit = False

        if unit.hit_info.hit:
            if str(unit.hit_info.entities[0]) == "target":
                target_hit = True
            else:
                target_hit = False

        for i in unit.vision:
            if unit.vision.index(i) <= (vision_range-1)/2:
                if i.hit:
                    if str(i.entities[0]) == "target":
                        left_target = True
                        continue
                    else:
                        left_target = False
                    if str(i.entities[0]) == "wall":
                        left_wall = True
                    else:
                        left_wall = False

            if unit.vision.index(i) >= (vision_range-1)/2:
                if i.hit:
                    if str(i.entities[0]) == "target":
                        right_target = True
                        continue
                    else:
                        right_target = False
                    if str(i.entities[0]) == "wall":
                        right_wall = True
                    else:
                        right_wall = False

            if unit.vision.index(i) == (vision_range-1)/2 + 1:
                if i.hit:
                    if str(i.entities[0]) == "wall":
                        center_wall = True
                    else:
                        center_wall = False
                    if str(i.entities[0]) == "target":
                        center_target = True
                    else:
                        center_target = False

        if swarm_count*target_count/(swarm_count + target_count) > 1:
            scale = swarm_count*target_count/(swarm_count + target_count)
        else:
            scale = 1

        for i in targets:
            if distance(unit, i) < 3 + base_scale / (15*scale):
                target_detect = 1
            else:
                target_detect = 0

        for j in walls:
            if distance(unit, j) < np.ceil(np.log10(base_scale)*1.5) + 2*pow(2, 0.5):
                wall_detect = 1
                break
            else:
                wall_detect = 0

        #if unit.rotation_y > 360:
        #    unit.rotation_y -= 360
        #if unit.rotation_y < 0:
        #    unit.rotation_y += 360

        state = [
            left_target,

            left_wall,

            center_target,

            center_wall,

            right_target,

            right_wall,

            target_detect,

            wall_detect,

            target_hit
        ]

        #  print(state)

        for i in range(len(state)):
            if type(state[i]) == float:
                state[i] = float(state[i])
            elif state[i]:
                state[i] = True
            else:
                state[i] = False

        #  print(np.asarray(state))
        return np.asarray(state)

    def set_reward(self, unit, targets, swarm, scale, time_limit, done):
        """
        Return the reward.
        The reward is determined as so:
        (During the simulation)
            - If a unit succeeds, it is instantly rewarded on the spot, then stops receiving any negative rewards
            - If a unit gets disabled, it stops receiving rewards
            - If a unit touches a wall or is stationary, without having succeeded, it's also punished
            - If a unit is in action, it will be punished if it increases its distance from its closest target,
              but is rewarded slightly for progressing towards its closest target
            - Positive rewards scale upwards based on how many targets have been found

        (At the end of the simulation)
            - If a unit succeeds, a reward is given to the swarm equal to the number of swarm units
            - if a unit fails, it's punished based on how far it is away from the nearest available target
            - The unit punishment won't ever exceed the number of successful units
            With this system for the end of a simulation, if one unit succeeds, everyone gets a reward, but then every
            single point is the responsibility of the other units to maintain, so the closer a unit is to a target, the
            more of the point it gets to retain.
            If two units succeed, then the remainder are responsible for double the number of points
        """
        self.reward = 0
        if not done:
            if unit.success:
                #self.reward = swarm*50/(len(targets) + 1)
                #self.reward += swarm*(1 - unit.time_lapsed/time_limit)
                self.reward = 1000
                print("###########################")
                print("unit succeeded and rewarded")
                print("###########################")
                unit.success = 0
                return self.reward  #*(swarm - len(targets) + 1) * 3

            # Min = scale  # variable to store the smallest distance unit has from a target
            # for i in targets:
            #    if Min > distance(unit, i):  # finding the smallest distance a unit has from a target
            #        Min = distance(unit, i)

            if not unit.success:
                #  self.reward = -0.0005*len(targets)

                ##  Rewarding for closer proximity to a target  ##
                #  if Min < scale/3:
                #      self.reward += 0.001*(swarm - len(targets) + 1)
                #  if Min < scale/6:
                #      self.reward += 0.0005*(swarm - len(targets) + 1)
                #  if Min < scale/10:
                #      self.reward += 0.0005*(swarm - len(targets) + 1)
                #  if Min < scale / 100:
                #      self.reward += 0.001*(swarm - len(targets) + 1)


                ##  Punishing for not making progress to a target  ##
                #  if unit.progress > Min:
                #    self.reward = 0.5 * (len(swarm) + 1 - len(targets))
                #    self.reward = 0
                #    unit.progress = Min
                #  elif unit.progress <= Min:
                #    self.reward = -0.1/(swarm + 1 - len(targets))

                # punishing stationary units, other unit contact and getting disabled
                if unit.stationary:
                    self.reward -= (time.time() - unit.stationary_start)
                if str(unit.hit_info.entity) == "unit":
                    self.reward -= 1
                if str(unit.hit_info.entity) == "wall":
                    self.reward -= 3
                if unit.disabled:
                    if unit.time_lapsed > 0.2:
                        self.reward -= 30
                    unit.disabled = False

        #else:
        #    minimum = scale
        #    if not unit.success or not unit.rewarded:
        #        for j in targets:
        #            if distance(j, unit) < minimum:
        #                minimum = distance(j, unit)
        #        self.reward = (swarm - len(targets))*((scale - minimum)/scale)*swarm

        return round(self.reward, 4)

    def remember(self, state, action, reward, next_state, done):
        """
        Store the <state, action, reward, next_state, is_done> tuple in a
        memory buffer for replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay_new(self, memory, batch_size):
        """
        Replay memory.
        """
        if len(memory) > batch_size:
            minibatch = random.sample(memory, batch_size)
        else:
            minibatch = memory
        for state, action, reward, next_state, done in minibatch:
            self.train()
            torch.set_grad_enabled(True)
            target = reward
            next_state_tensor = torch.tensor(np.expand_dims(next_state, 0), dtype=torch.float32).to(DEVICE)
            state_tensor = torch.tensor(np.expand_dims(state, 0), dtype=torch.float32, requires_grad=True).to(DEVICE)
            if not done:
                target = reward + self.gamma * torch.max(self.forward(next_state_tensor)[0])
            output = self.forward(state_tensor)
            target_f = output.clone()
            target_f[0][np.argmax(action)] = target
            target_f.detach()
            self.optimizer.zero_grad()
            loss = F.mse_loss(output, target_f)
            loss.backward()
            self.optimizer.step()

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the DQN agent on the <state, action, reward, next_state, is_done>
        tuple at the current timestep.
        """
        self.train()
        torch.set_grad_enabled(True)
        target = reward
        next_state_tensor = torch.tensor(next_state.reshape((1, 9)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 9)), dtype=torch.float32, requires_grad=True).to(DEVICE)
        if not done:
            target = reward + self.gamma * torch.max(self.forward(next_state_tensor[0]))
        output = self.forward(state_tensor)
        target_f = output.clone()
        target_f[0][np.argmax(action)] = target
        target_f.detach()
        self.optimizer.zero_grad()
        loss = F.mse_loss(output, target_f)
        loss.backward()
        self.optimizer.step()
