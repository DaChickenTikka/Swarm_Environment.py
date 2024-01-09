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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class SwarmTargetDQNAgent(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = None
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
        self.optimizer = optim.Adam(self.parameters(), weight_decay=0, lr=params['learning_rate'])
        self.network()

    def network(self):
        # Layers
        self.f1 = nn.Linear(7, self.first_layer)
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
            if unit.vision.index(i) <= (vision_range - 1) / 2:
                if i.hit:
                    if str(i.entities[0])[0] == "T":
                        left_target = True
                        continue
                    else:
                        left_target = False
                    if str(i.entities[0])[0] == "W":
                        left_wall = True
                    else:
                        left_wall = False

            if unit.vision.index(i) >= (vision_range - 1) / 2:
                if i.hit:
                    if str(i.entities[0])[0] == "T":
                        right_target = True
                        continue
                    else:
                        right_target = False
                    if str(i.entities[0])[0] == "W":
                        right_wall = True
                    else:
                        right_wall = False

            if unit.vision.index(i) == (vision_range - 1) / 2 + 1:
                if i.hit:
                    if str(i.entities[0])[0] == "W":
                        center_wall = True
                    else:
                        center_wall = False
                    if str(i.entities[0])[0] == "T":
                        center_target = True
                    else:
                        center_target = False

        if unit.detect.hit:
            if str(unit.detect.entities[0])[0] == "T":
                target_detect = 1
            else:
                target_detect = 0
            if str(unit.detect.entities[0])[0] == "W":
                wall_detect = 1
            else:
                wall_detect = 0
        else:
            target_detect = 0
            wall_detect = 0

        state = [

            # target_detect,

            left_target,

            center_target,

            right_target,

            wall_detect,

            left_wall,

            center_wall,

            right_wall
        ]

        #  print(state)

        for i in range(len(state)):
            if state[i]:
                state[i] = True
            else:
                state[i] = False

        # print(np.asarray(state))
        return np.asarray(state)

    def set_reward(self, unit, targets, swarm, scale):
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
        if unit.success:
            self.reward = 1  # 20*(int(unit.target_count/2) <= 1) + 50*(2**(int(unit.target_count/2) - 1))*(int(unit.target_count/2) >= 2)
            print("###################################################")
            print(f"{unit.name} succeeded and rewarded, total targets {unit.target_count}")
            print("###################################################")
            unit.success = 0
            return self.reward  # *(swarm - len(targets) + 1) * 3

        if not unit.success:

            # punishing stationary units, contact with other units and walls, rewarding movement slightly
            if not unit.stationary:
                self.reward = 0  # b0.001  #(time.time() - unit.stationary_start)*2
            else:
                self.reward = -0.0001
            if unit.detect.hit:
                if str(unit.detect.entities[0])[0] == "U":
                    self.reward -= 0.0001
                if str(unit.detect.entities[0])[0] == "W":
                    self.reward -= 0.0001
            # if unit.goal_found:
            #    self.reward += (2**(unit.target_count/2))*0.1
            #    print("########### Found mini goal ###########")
            #    unit.goal_found = 0
            # if unit.goal_lost:
            #    self.reward -= 0.1*1/(unit.target_count/2 + 1)
            #    print("############ Losing target ############")
            #    unit.goal_lost = 0
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
        next_state_tensor = torch.tensor(next_state.reshape((1, 7)), dtype=torch.float32).to(DEVICE)
        state_tensor = torch.tensor(state.reshape((1, 7)), dtype=torch.float32, requires_grad=True).to(DEVICE)
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

    '''def act(self, state, eps):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state.reshape((1, 7)), dtype=torch.float32).to(
                DEVICE)
        prediction = agent(state_tensor)
        final_move = argmax(prediction.detach().cpu().numpy())

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))



class DuelingDQN(nn.Module):
  #  """Convolutional neural network for the Atari games."""

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
    '''


# credit for set up below: https://github.com/surajitsaikia27/Deep-Reinforcement-Learning-for-Navigation/blob/master/dqn_agent.py
# import numpy as np
# import random
# import torch
# import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64  # minibatch size
GAMMA = 0.995  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 5e-4  # learning rate
UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1=32, fc2=64, fc3=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # "*** YOUR CODE HERE ***"
        self.fc1 = nn.Linear(state_size, fc1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(fc1, fc2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(fc2, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        state = self.fc1(state)
        state = self.relu1(state)
        state = self.fc2(state)
        state = self.relu2(state)
        state = self.fc3(state)

        return state


class DDQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, params):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=params['learning_rate'])
        # Replay memory
        self.memory = ReplayBuffer(action_size, params['memory_size'], params['batch_size'], seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.epsilon = 0
        self.reward = 0


    def step(self, state, action, reward, next_state, done, params):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > params['batch_size']:
                experiences = self.memory.sample()
                self.learn(experiences, params['gamma'], params['tau'], params['batch_size'], params['learning_rate'])

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)  # same as self.qnetwork_local.forward(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, tau, batch_size, learning_rate):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # "*** YOUR CODE HERE ***"
        qs_local = self.qnetwork_local.forward(states)
        qsa_local = qs_local[torch.arange(batch_size, dtype=torch.long), actions.reshape(batch_size)]
        qsa_local = qsa_local.reshape((batch_size, 1))
        # print(qsa_local.shape)

        # # DQN Target
        # qs_target = self.qnetwork_target.forward(next_states)
        # qsa_target, _ = torch.max(qs_target, dim=1) #using the greedy policy (q-learning)
        # qsa_target = qsa_target * (1 - dones.reshape(BATCH_SIZE)) #target qsa value is zero when episode is complete
        # qsa_target = qsa_target.reshape((BATCH_SIZE,1))
        # TD_target = rewards + gamma * qsa_target
        # #print(qsa_target.shape, TD_target.shape, rewards.shape)

        # # Double DQN Target ver 1
        # qs_target = self.qnetwork_target.forward(next_states)
        # if random.random() > 0.5:
        #     _, qsa_target_argmax_a = torch.max(qs_target, dim=1) #using the greedy policy (q-learning)
        #     qsa_target = qs_target[torch.arange(BATCH_SIZE, dtype=torch.long), qsa_target_argmax_a.reshape(BATCH_SIZE)]
        # else:
        #     _, qsa_local_argmax_a = torch.max(qs_local, dim=1) #using the greedy policy (q-learning)
        #     #qsa_target = qs_target[torch.arange(BATCH_SIZE, dtype=torch.long), qsa_local_argmax_a.reshape(BATCH_SIZE)]
        #     ##qsa_target = qs_local[torch.arange(BATCH_SIZE, dtype=torch.long), qsa_local_argmax_a.reshape(BATCH_SIZE)]

        # qsa_target = qsa_target * (1 - dones.reshape(BATCH_SIZE)) #target qsa value is zero when episode is complete
        # qsa_target = qsa_target.reshape((BATCH_SIZE,1))
        # TD_target = rewards + gamma * qsa_target

        # Double DQN Target ver 2 (based upon double dqn paper)
        qs_target = self.qnetwork_target.forward(next_states)
        _, qsa_local_argmax_a = torch.max(qs_local, dim=1)  # using the greedy policy (q-learning)
        qsa_target = qs_target[torch.arange(batch_size, dtype=torch.long), qsa_local_argmax_a.reshape(batch_size)]

        qsa_target = qsa_target * (1 - dones.reshape(batch_size))  # target qsa value is zero when episode is complete
        qsa_target = qsa_target.reshape((batch_size, 1))
        TD_target = rewards + gamma * qsa_target

        # print(qsa_target.shape, TD_target.shape, rewards.shape)

        # #Udacity's approach
        # # Get max predicted Q values (for next states) from target model
        # Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # # Compute Q targets for current states
        # TD_target = rewards + (gamma * Q_targets_next * (1 - dones))
        # # Get expected Q values from local model
        # qsa_local = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(qsa_local, TD_target)  # much faster than the above loss function
        # print(loss)
        # minimize the loss
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate
        self.optimizer.zero_grad()  # clears the gradients
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        weights = []

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
