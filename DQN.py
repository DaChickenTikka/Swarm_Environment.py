import random
import numpy as np
import pandas as pd
# from operator import add
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
# import time
# import torch.optim as optim
# import copy
import math
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SwarmTargetDQNAgent(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.reward = 0
        self.gamma = 0.99
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
        self.f1 = nn.Linear(8, self.first_layer)
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
        The state is a numpy array of 8 values, representing:
            - 0, 1: Rays left see entity => target or wall
            - 2, 3: Center ray sees entity => target or wall
            - 4, 5: Rays right see entity => target or wall
            - target proximity detection
            - wall proximity detection
        """

        vision_range = unit.vision_lines
        left_target = False
        left_wall = False
        right_target = False
        right_wall = False
        center_target = False
        center_wall = False

        for i in unit.vision:
            if unit.vision.index(i) <= (vision_range-1)/2:
                if i.hit:
                    if str(i.entities[0]) in targets:
                        left_target = True
                        continue
                    else:
                        left_target = False
                    if str(i.entities[0]) in walls:
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

        if unit.detect_entity.hit:
            detected_set = {str(df.name) for df in unit.detect_entity.entities}
            if detected_set.intersection({str(tar) for tar in targets}) != set():
                target_detect = 1
            else:
                target_detect = 0
            if detected_set.intersection({str(wall) for wall in walls}) != set():
                wall_detect = 1
            else:
                wall_detect = 0
        else:
            target_detect = 0
            wall_detect = 0

        state = [
            left_target,

            left_wall,

            center_target,

            center_wall,

            right_target,

            right_wall,

            target_detect,

            wall_detect
        ]

        #  print(state)

        for i in range(len(state)):
            if state[i]:
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
        """
        self.reward = 0
        if not done:
            if unit.success:
                self.reward = 20*(int(unit.target_count/2) <= 1) + 100*(2**(int(unit.target_count/2) - 1))*(int(unit.target_count/2) >= 2)
                print("###################################################")
                print(f"{unit.name} succeeded and rewarded, total targets {unit.target_count/2}")
                print("###################################################")
                unit.success = 0
                return self.reward  #*(swarm - len(targets) + 1) * 3

            if not unit.success:

                # punishing stationary units, other unit contact and getting disabled, rewarding movement slightly
                if not unit.stationary:
                    self.reward = 0.001  #(time.time() - unit.stationary_start)*2
                else:
                    self.reward = -0.001
                if str(unit.hit_info.entity)[0] == "U":
                    self.reward -= 0.8
                if str(unit.hit_info.entity)[0] == "W":
                    self.reward -= 1.5
                if unit.goal_found:
                    self.reward += (2**(unit.target_count/2))*0.1
                    print("########### Found mini goal ###########")
                    unit.goal_found = 0
                if unit.goal_lost:
                    self.reward -= 0.5*1/(unit.target_count/2 + 1)
                    print("############ Losing target ############")
                    unit.goal_lost = 0
                if unit.disabled == 1:
                    unit.disabled = 0
                    # self.reward -= 500
                    print("########### Unit Disabled ###########")

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
        next_state_tensor = torch.tensor(next_state.reshape((1, 8)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 8)), dtype=torch.float32, requires_grad=True).to(DEVICE)
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



class DuelingDQN(torch.nn.Module):
    """Convolutional neural network for the Atari games."""

    def __init__(self, num_actions):
        super(DuelingDQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        std = math.sqrt(2.0 / (4 * 84 * 84))
        nn.init.normal_(self.conv1.weight, mean=0.0, std=std)
        self.conv1.bias.data.fill_(0.0)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        std = math.sqrt(2.0 / (32 * 4 * 8 * 8))
        nn.init.normal_(self.conv2.weight, mean=0.0, std=std)
        self.conv2.bias.data.fill_(0.0)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        std = math.sqrt(2.0 / (64 * 32 * 4 * 4))
        nn.init.normal_(self.conv3.weight, mean=0.0, std=std)
        self.conv3.bias.data.fill_(0.0)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        std = math.sqrt(2.0 / (64 * 64 * 3 * 3))
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        self.fc1.bias.data.fill_(0.0)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, num_actions)

    def forward(self, states):
        """Forward pass of the neural network with some inputs."""
        x = F.relu(self.conv1(states))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))  # Flatten imathut.
        V = self.V(x)
        A = self.A(x)
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q


