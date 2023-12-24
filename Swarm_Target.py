#  27th May 2023, design decision made, blocks can rotate on the spot and only move forward, happy birthday to myself I guess XD
from ursina import *
from cmath import rect
from numpy import *  # real, imag, pi, array, sin, cos, random
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from DQN import SwarmTargetDQNAgent
from random import randint
import random
import statistics
import torch.optim as optim
import torch
#  from GPyOpt.methods import BayesianOptimization
from bayesOpt import *
import datetime
import distutils.util

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using " + DEVICE)


def Real(d):
    c = array([d, ])
    return real(c)


def Imag(p):
    c = array([p, ])
    return imag(c)


def define_parameters():
    paramse = dict()
    # Spectator Settings
    paramse["view_mode"] = "P"       #[F]ull view, [P]artial view
    # Environment
    paramse["time_limit"] = 10
    paramse["base_scale"] = 20
    paramse["swarm_size"] = 1
    paramse["targets_amount"] = 1
    # Neural Network
    episodes = 1500
    paramse['first_layer_size'] = 81  # neurons in the first layer
    paramse['second_layer_size'] = 93  # neurons in the second layer
    paramse['third_layer_size'] = 81  # neurons in the third layer
    paramse['learning_rate'] = 0.007
    paramse['epsilon_decay'] = 0.02 ** (3 / (2*episodes))
    paramse['memory_size'] = 16384
    paramse['batch_size'] = 1024
    paramse['episodes'] = episodes
    paramse['episode_count'] = 0
    paramse['time_lapsed'] = 0
    # Settings
    paramse['load_weights'] = False
    paramse['train'] = True
    paramse["test"] = False
    paramse["continue_train"] = False
    paramse['plot_score'] = True
    ID = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    if paramse["train"]:
        ID += '_train_'
    if paramse["test"]:
        ID += '_test_'
    paramse['weights_path'] = 'weights/' + ID + '.h5'
    paramse['log_path'] = 'logs/scores' + ID + '.txt'
    return paramse


class Target(Entity):
    def __init__(self, add_to_scene_entities=True, **kwargs):
        super().__init__(add_to_scene_entities, **kwargs)
        self.disabled = 0
        self.hit_info = self.intersects()
        self.name = 0

    def update(self):
        self.rotation_y += 100 * time.dt
        self.hit_info = self.intersects()
        base_scale = params["base_scale"]

        if self.hit_info.hit:
            if str(self.hit_info.entity) in swarm:
                self.hit_info.entities[0].target_count += 1
                self.hit_info.entities[0].success = 1

        if (self.x == 0 and self.y == 0) or len(targets) > target_count:
            self.disabled = 1

        if self.disabled or str(self.hit_info.entity) in swarm:
            while 1:
                safe = True
                sx = random.randint(round(-(base_scale - 1) / 2) + 4, round((base_scale - 1) / 2) - 4)
                sz = random.randint(round(-(base_scale - 1) / 2) + 4, round((base_scale - 1) / 2) - 4)
                for a in targets:
                    if targets[a].x + 2 > sx > targets[a].x - 2 and targets[a].z + 2 > sz > targets[a].z - 2:
                        safe = False
                for a in swarm:
                    if swarm[a].x + 2 > sx > swarm[a].x - 2 and swarm[a].z + 2 > sz > swarm[a].z - 2:
                        safe = False
                if safe:
                    break
            self.x = sx
            self.z = sz
            self.disabled = 0

        if len(targets) < target_count:
            while 1:
                safe = True
                sx = random.randint(round(-(base_scale - 1) / 2) + 3, round((base_scale - 1) / 2) - 3)
                sz = random.randint(round(-(base_scale - 1) / 2) + 3, round((base_scale - 1) / 2) - 3)
                for a in targets:
                    if targets[a].x + 2 > sx > targets[a].x - 2 and targets[a].z + 2 > sz > targets[a].z - 2:
                        safe = False
                for a in swarm:
                    if swarm[a].x + 2 > sx > swarm[a].x - 2 and swarm[a].z + 2 > sz > swarm[a].z - 2:
                        safe = False
                if safe:
                    break
            self.x = sx
            self.z = sz
            targets["T" + str(len(targets))] = Target(name="T" + str(len(targets)), model='cube', texture='vignette',
                                           # scale=(0.75, 0.75, 0.75),
                                           collider='box', origin_y=-.5, color=color.violet,
                                           x=sx,
                                           y=0.1,
                                           z=sz)
            targets["T" + str(len(targets)-1)].name = "T" + str(len(targets)-1)


class Wall(Entity):
    def __init__(self, add_to_scene_entities=True, **kwargs):
        super().__init__(add_to_scene_entities, **kwargs)


class Unit(Entity):

    def __init__(self, add_to_scene_entities=True, **kwargs):
        super().__init__(add_to_scene_entities, **kwargs)
        self.rotation_y = 0
        self.direction = (Vec3(sin(self.rotation_y * pi / 180), 0, cos(self.rotation_y * pi / 180)))
        self.position = self.world_position
        self.movement = True
        self.success = False
        self.vision = []
        self.vision_lines = 7
        self.len_vision = params["base_scale"] * 0.5
        self.move = [0, 0, 0]  # [turn left, go forward, turn right]
        self.choice = 0
        self.previous_position = self.world_position
        self.stationary = False
        self.stationary_start = time.time()
        self.rewarded = 0
        self.disabled = 0
        self.reward = 0
        self.respawn = False
        self.target_count = 0
        self.out_of_range = 0
        self.max_distance = 0
        self.prev_min = params['base_scale']
        self.goal_found = 0
        self.goal_lost = 0
        self.name = "U0"
        boundary = ceil(2*log10(params["base_scale"]))/1.5
        self.detect = Entity(name="detect", parent=self, collider="box", position=Vec3(0, 0, 0), scale=(boundary, 0.1, boundary), color=color.azure)
        self.detect.parent = self
        self.detect_entity = self.detect.intersects(ignore=(self,))
        self.hit_info = self.intersects(ignore=(self.detect,))

        origin = self.world_position + (self.up * 0.1)

        if self.vision_lines % 2 == 0:
            self.vision_lines += 1
        q = 0
        while q < self.vision_lines:
            if self.vision_lines == 1:
                dir = self.direction
            else:  # set the new outer bound
                reference_angle = self.rotation_y * pi / 180 + pi * 3 / 4
                difference = (pi / 2) * (q / (self.vision_lines - 1))
                direction = rect(1, reference_angle - difference)
                dirx = Real(direction)[0]
                dirz = Imag(direction)[0]
                dir = Vec3(-dirx, 0, dirz).normalized()

            self.vision.append(
                raycast(origin, direction=dir, ignore=(self, Unit, Wall, self.detect), distance=self.len_vision, debug=False))
            q += 1

    def update(self):
        base_scale = params["base_scale"]
        #  controlling spawn behaviour and size and eliminating swarm if it somehow passes the boundaries

        if self.x > round((base_scale + 1) / 2) - 0.35 or self.x < -round((base_scale + 1) / 2) + 0.35 or \
                self.z > round((base_scale + 1) / 2) - 0.35 or self.z < -round((base_scale + 1) / 2) + 0.35:
            self.disabled = 1
            while 1:
                safe = True
                sx = random.randint(round(-(base_scale - 1) / 2) + 3, round((base_scale - 1) / 2) - 3)
                sz = random.randint(round(-(base_scale - 1) / 2) + 3, round((base_scale - 1) / 2) - 3)
                for a in targets:
                    if targets[a].x + 2 > sx > targets[a].x - 2 and targets[a].z + 2 > sz > targets[a].z - 2:
                        safe = False
                if safe:
                    break
            self.x = sx
            self.z = sz
            self.look_at(target=targets["T0"])
            temp_rotation = self.rotation_y
            self.rotation_y = temp_rotation + 45*(float(round((params['episode_count']/params['episodes'])**4, 2)))*((-1)**params['episode_count'])

        '''
                i = len(swarm)
                while i < swarm_count:
                    while 1:
                        safe = True
                        sx = random.randint(round(-(base_scale - 1) / 2) + 3, round((base_scale - 1) / 2) - 3)
                        sz = random.randint(round(-(base_scale - 1) / 2) + 3, round((base_scale - 1) / 2) - 3)
                        for a in targets:
                            if targets[a].x + 2 > sx > targets[a].x - 2 and targets[a].z + 2 > sz > targets[a].z - 2:
                                safe = False
                        if safe:
                            break
                    swarm["U" + str(i)].look_at(target=targets["T0"])

                    i = len(swarm)
        '''

        # encouragement for the unit to explore more if it's not able to see a target
        if params['train'] and distance(self, targets[str(max(targets))]) > 0:  # self.len_vision/10:
            dist = distance(self, targets[str(max(targets))])

            # reward unit for approaching target
            if dist > self.max_distance:
                self.max_distance = dist

            if dist <= 5 * self.max_distance / 6:
                self.goal_found = 1
                self.max_distance = 0

            # punish unit for leaving target
            if dist < self.prev_min:
                self.prev_min = dist

            if dist**2 >= (6/5)*(self.prev_min**2):  # using exponents to increase punishment frequency the further distance away is found
                self.goal_lost = 1
                self.prev_min = dist


        if not self.movement:
            pass
        else:
            # rotation control
            self.rotation_y += (self.move[2] - self.move[0]) * time.dt * 220

            # direction control
            self.direction = (Vec3(sin(self.rotation_y * pi / 180), 0, cos(self.rotation_y * pi / 180)))
            self.detect.rotation_y = self.rotation_y
            # the ray should start slightly up from the ground, so we can walk up slopes or walk over small objects.
            origin = self.world_position + (self.up * 0.1)

            # hit info based on intersection
            self.hit_info = self.intersects(ignore=(self.detect, ))
            self.detect_entity = self.detect.intersects(ignore=(self, ))
            # print("Current position: " + str(self.world_position) + ", detect position: " + str(self.detect.world_position) + "Detected entities: " + str(self.detect_entity.entities))
            self.previous_position = self.position  # save the current position of the cube before it updates

            # if the ray doesn't hit anything, allow movement
            if str(self.hit_info.entity) not in swarm and "W" not in str(self.hit_info.entity):
                self.position += self.direction * params["base_scale"] / 4 * time.dt * self.move[1]
                # self.detect.position = self.position
                # self.detect.rotation_y = self.rotation_y

            # if the unit hits a target, immediately teleport the target, set the cube to have succeeded, then
            # bring the target into a new random location
            elif str(self.hit_info.entity) == "target":
                self.success = 1
                # self.movement = False
                # self.color = color.white
                targets[self.hit_info.entity.name].respawn = 0
                targets[self.hit_info.entity.name].disabled = 1

            # check if the cube is stationary
            if self.position == self.previous_position:
                self.stationary = True
            # self.move_start = time.time()
            else:
                self.stationary_start = time.time()
                self.stationary = False
            # if a success metric has been achieved, do the following:
            if self.success:
                self.target_count += 1

            # if success hasn't been achieved, then allow the cube to refresh its vision rays
            if not self.success:
                # setting up the vision of the swarm unit
                x = 0
                while x < self.vision_lines:
                    if self.vision_lines == 1:
                        dir = self.direction
                    else:
                        reference_angle = self.rotation_y * pi / 180 + pi * 3 / 4
                        difference = (pi / 2) * (x / (self.vision_lines - 1))
                        direction = rect(1, reference_angle - difference)
                        dirx = Real(direction)[0]
                        dirz = Imag(direction)[0]
                        dir = Vec3(-dirx, 0, dirz).normalized()
                    self.vision[x] = raycast(origin, dir, ignore=(self, self.detect), distance=self.len_vision, debug=False)
                    x += 1

                for v in self.vision:
                    if not v.hit:
                        self.color = color.blue
                        if self.detect_entity:
                            self.color = color.white
                        else:
                            self.color = color.blue
                    else:
                        # print(f"{self.position} sees first {v.entity}, total {v.entities}")
                        if str(v.entity) in targets:
                            self.color = color.red
                            break
                        else:
                            self.color = color.blue
                            if self.detect_entity:
                                self.color = color.white
                            else:
                                self.color = color.blue


class Camera(Entity):  # The camera for viewing this project in action
    def __init__(self):
        super().__init__()
        self.height = 0.1
        self.camera_pivot = Entity(parent=self, y=self.height)
        camera.parent = self.camera_pivot
        camera.position = (0, 0, 0)
        camera.rotation = (0, 0, 0)
        camera.fov = 90
        mouse.locked = True
        self.mouse_sensitivity = Vec2(40, 40)
        self.speed = 10
        self.direction = Vec3(0, 0, 0)

    def update(self):
        self.rotation_y += mouse.velocity[0] * self.mouse_sensitivity[1]
        self.camera_pivot.rotation_x -= mouse.velocity[1] * self.mouse_sensitivity[0]
        self.camera_pivot.rotation_x = clamp(self.camera_pivot.rotation_x, -90, 90)

        self.direction = Vec3(
            self.forward * (held_keys['w'] - held_keys['s']) * self.speed
            + self.up * (held_keys['q'] - held_keys['e']) * self.speed
            + self.right * (held_keys['d'] - held_keys['a']) * self.speed
        ).normalized()  # get the direction we're view to walk in.

        # the ray should start slightly up from the ground, so we can walk up slopes or walk over small objects.
        origin = self.world_position + (
                self.up * 0.5)
        hit_info = raycast(origin, self.direction, ignore=(self, Entity, Unit), distance=0.01,
                           debug=False)
        if not hit_info.hit:
            if held_keys['c']:
                self.position += self.direction * 3 * time.dt
            else:
                self.position += self.direction * 10 * time.dt


def plot_seaborn(array_counter, array_score, train):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13, 8))
    fit_reg = False if not train else True
    ax = sns.regplot(
        x=array([array_counter]),
        y=array([array_score]),
        # color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg=fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [mean(array_score)] * len(array_counter)
    ax.plot(array_counter, y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='Episode', ylabel='Total Reward')
    plt.savefig(fname='plots/' + params['log_path'].lstrip('logs/scores_').rstrip(".txt") + '_plot.png', format='png')


def get_mean_stdev(arr):
    return statistics.mean(arr), statistics.stdev(arr)


def test(paramss):
    paramss['load_weights'] = True
    paramss['train'] = False
    paramss["test"] = True
    score = run_simulation(paramss)
    return score


done = 0


def run_simulation(params):
    window.vsync = False

    match params["view_mode"]:

        # Full View
        case "F":
            app = Ursina(size=(1800, 900))
            timer = Text(text="", x=-0.98, y=0.44)  # setting up a timer to record and display the time of the simulation
            Target_count = Text(text='', x=-0.98, y=0.4)
            swarm_counter = Text(text='', x=-0.98, y=0.36)
            episode_count = Text(text='', x=-0.98, y=0.48)
            train_test = Text(text='', x=0.90, y=0.44)

        # Partial View
        case "P":
            app = Ursina(size=(1000, 650), position=(920, 0))
            timer = Text(text="", x=-0.73, y=0.44)  # setting up a timer to record and display the time of the simulation
            Target_count = Text(text='', x=-0.73, y=0.40)
            swarm_counter = Text(text='', x=-0.73, y=0.36)
            episode_count = Text(text='', x=-0.73, y=0.48)
            train_test = Text(text='', x=0.63, y=0.472)

        # default case, Full View
        case _:
            app = Ursina(size=(1800, 900))
            timer = Text(text="", x=-0.98, y=0.44)  # setting up a timer to record and display the time of the simulation
            Target_count = Text(text='', x=-0.98, y=0.4)
            swarm_counter = Text(text='', x=-0.98, y=0.36)
            episode_count = Text(text='', x=-0.98, y=0.48)
            train_test = Text(text='', x=0.55, y=0.44)

    #  set the size scaling you want for the world
    base_scale = params["base_scale"]

    # set the time limit in seconds
    # noinspection PyGlobalUndefined
    global time_limit
    time_limit = params["time_limit"]

    # the ground for the swarm to work from
    ground = Entity(model='plane', scale=(base_scale, 1, base_scale), color=color.yellow.tint(-.2),
                    texture='white_cube',
                    texture_scale=(base_scale, base_scale), collider='box')
    ground.eternal = True

    timer.eternal = True
    Target_count.eternal = True
    swarm_counter.eternal = True
    episode_count.eternal = True
    train_test.eternal = True

    walls = {}

    # set the number of targets and/or swarm units you want
    global target_count
    target_count = params["targets_amount"]
    global swarm_count
    swarm_count = params["swarm_size"]

    wall_generate = 0

    while wall_generate < base_scale:
        walls["WN" + str(wall_generate)] = Wall(name="WN" + str(wall_generate), model='cube', collider='box',
                                    x=base_scale / 2 - wall_generate - 0.5, y=0.5, z=base_scale / 2 + 0.5,
                                    color=color.white10)
        walls["WS" + str(wall_generate)] = Wall(name="WS" + str(wall_generate), model='cube', collider='box',
                                    x=base_scale / 2 - wall_generate - 0.5, y=0.5, z=-base_scale / 2 - 0.5,
                                    color=color.white10)
        walls["WE" + str(wall_generate)] = Wall(name="WE" + str(wall_generate), model='cube', collider='box',
                                    x=base_scale / 2 + 0.5, y=0.5, z=base_scale / 2 - wall_generate - 0.5,
                                    color=color.white10)
        walls["WW" + str(wall_generate)] = Wall(name="WW" + str(wall_generate), model='cube', collider='box',
                                    x=-base_scale / 2 - 0.5, y=0.5, z=base_scale / 2 - wall_generate - 0.5,
                                    color=color.white10)
        wall_generate += 1

    for i in walls:
        walls[i].eternal = True

    # initialise the camera to view the world
    viewer = Camera()
    viewer.eternal = True
    viewer.y = 15
    viewer.z = -25

    agent = SwarmTargetDQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])

    define_parameters()

    if params['continue_train']:
        params['episode_count'] = 400
    else:
        params['episode_count'] = 0
    reward_plot = []
    time_plot = []


    while params['episode_count'] <= params['episodes']:
        # resetting the targets and swarm units for a new round of training
        scene.clear()
        total_reward = 0
        started = False

        # initialise the lists that will contain the swarm and target objects
        global swarm
        swarm = {}
        global targets
        targets = {}

        if params['train'] or params['continue_train']:
            train_test.text = 'Training'

        if params['test']:
            train_test.text = ' Testing'

        # walls_count = 25

        # place the targets and swarm units in random locations across the worlds
        j = len(targets)
        while j < target_count:
            targets["T" + str(j)] = Target(name="T" + str(j), model='cube', texture='vignette',  # scale=(0.75, 0.75, 0.75),
                                collider='box', origin_y=-.5,
                                color=color.violet,
                                x=random.randint(round(-(base_scale - 1) / 2) + 4, round((base_scale - 1) / 2) - 4),
                                y=0.1,
                                z=random.randint(round(-(base_scale - 1) / 2) + 4, round((base_scale - 1) / 2) - 4))
            targets["T" + str(j)].name = "T" + str(j)
            targets["T" + str(j)].targets_found = 0
            j = len(targets)

        i = len(swarm)
        while i < swarm_count:
            while 1:
                safe = True
                sx = random.randint(round(-(base_scale - 1) / 2) + 3, round((base_scale - 1) / 2) - 3)
                sz = random.randint(round(-(base_scale - 1) / 2) + 3, round((base_scale - 1) / 2) - 3)
                for a in targets:
                    if targets[a].x + 2 > sx > targets[a].x - 2 and targets[a].z + 2 > sz > targets[a].z - 2:
                        safe = False
                if safe:
                    break
            swarm["U" + str(i)] = Unit(name="U" + str(i), model='cube', collider='box', origin_y=-.5, color=color.blue,
                            x=sx, y=0.1, z=sz)
            swarm["U" + str(i)].name = "U" + str(i)
            swarm["U" + str(i)].look_at(target=targets["T0"])
            temp_rotation = swarm["U" + str(i)].rotation_y
            swarm["U" + str(i)].rotation_y = temp_rotation + 45*(float(round((params['episode_count']/params['episodes'])**4, 2)))*((-1)**params['episode_count'])

            i = len(swarm)

        global start_time
        start_time = time.time()

        if not params['train']:
            agent.epsilon = 0.02
        else:
            # agent.epsilon is set to give randomness to actions
            agent.epsilon = power(params["epsilon_decay"], params['episode_count'])
            if agent.epsilon < 0.02:
                agent.epsilon = 0.02

        print(f"Start of episode {params['episode_count']}")
        done = 0
        params['time_lapsed'] = 0
        app.step()
        while params['time_lapsed'] < params['time_limit']:  # and len(swarm) > 0:

            # set the start time at the start of the time loop once, then don't update again until next episode
            if not started:
                # set the start time as the beginning of the game
                start_time = time.time()
                started = True
            else:
                total_targets = 0
                for t in swarm:
                    total_targets += swarm[str(t)].target_count

                params['time_lapsed'] = time.time() - start_time  # subtract the current time from the start time for the time elapsed
                timer.text = "Time: " + str(round(params['time_lapsed'], 3)) + 's'  # print the timer
                swarm_counter.text = "Units Remaining: " + str(len(swarm))  # print the target counter
                episode_count.text = "Episode " + str(params['episode_count'])  # print the episode counter
                Target_count.text = "Targets Found: " + str(int(total_targets/2))  # print the target counter

                # get old state
                for i in swarm:
                    state_old = agent.get_state(swarm[i], targets, walls, swarm_count, target_count, base_scale)

                    # perform random actions based on agent.epsilon, or choose the action
                    swarm[i].choice = random.uniform(0, 1)
                    if swarm[i].choice < agent.epsilon:
                        final_move = eye(3)[randint(0, 2)]
                    else:
                        # predict action based on the old state
                        with torch.no_grad():
                            state_old_tensor = torch.tensor(state_old.reshape((1, 8)), dtype=torch.float32).to(
                                DEVICE)
                            prediction = agent(state_old_tensor)
                            final_move = eye(3)[argmax(prediction.detach().cpu().numpy()[0])]
                        # perform new move and get new state
                    swarm[i].move[0] = final_move[0]
                    swarm[i].move[1] = final_move[1]
                    swarm[i].move[2] = final_move[2]
                app.step()
                for i in swarm:
                    # print(f"{i}, location {swarm[i].world_position}")
                    state_new = agent.get_state(swarm[i], targets, walls, swarm_count, target_count, base_scale)
                    # set reward for the new state
                    Reward = agent.set_reward(swarm[i], targets, swarm_count, base_scale, time_limit, done)
                    swarm[i].reward = Reward
                    # print(f"unit: {i}, hit: {swarm[i].hit_info.entity}, detect: {swarm[i].detect_entity.entity}, move: {swarm[i].move}, reward: {swarm[i].reward}, epsilon: {agent.epsilon}, choice: {swarm[i].choice}")
                    total_reward += Reward

                    if params['train'] or params['continue_train']:
                        # train short memory base on the new action and state
                        agent.train_short_memory(state_old, final_move, Reward, state_new, done)
                        # store the new data into a long term memory
                        agent.remember(state_old, final_move, Reward, state_new, done)

                        model_weights = agent.state_dict()
                        torch.save(model_weights, params["weights_path"])

        done = 1
        print("####################")
        print("   End of Episode")
        print("####################")
        for i in swarm:
            state_new = agent.get_state(swarm[i], targets, walls, swarm_count, target_count, base_scale)
            Reward = agent.set_reward(swarm[i], targets, swarm_count, base_scale, time_limit, done)
            # print(f"End of Episode Reward: {Reward}")
            if params['train'] or params['continue_train']:
                # train short memory base on the new action and state
                agent.train_short_memory(state_old, final_move, Reward, state_new, done)
                # store the new data into a long term memory
                agent.remember(state_old, final_move, Reward, state_new, done)
            total_reward += Reward

        if params['train'] or params['continue_train']:
            model_weights = agent.state_dict()
            torch.save(model_weights, params["weights_path"])
        reward_plot.append(total_reward)
        time_plot.append(params['episode_count']/2)

        print(f"Total Episode Reward: ({total_reward})")
        if params['train'] or params['continue_train']:
            agent.replay_new(agent.memory, params['batch_size'])
        params['episode_count'] += 1
        scene.clear()
        app.step()

        if params['plot_score'] and params['episode_count'] % 15 == 0:
            plot_seaborn(time_plot, reward_plot, params['train'])

    if params['plot_score']:
        plot_seaborn(time_plot, reward_plot, params['train'])
    meaN, stdev = get_mean_stdev(reward_plot)

    return total_reward, meaN, stdev


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--bayesianopt", nargs='?', type=distutils.util.strtobool, default=False)  # not params['test'])
    args = parser.parse_args()
    print("Args", args)
    params['display'] = args.display
    if args.bayesianopt:
        bayesOpt = BayesianOptimizer(params)
        bayesOpt.optimize_RL()
    if params['train']:
        print("Training...")
        params['load_weights'] = False  # when training, the network is not pre-trained
        run_simulation(params)
    if params['continue_train']:
        params['load_weights'] = True
        run_simulation(params)
    if params['test']:
        print("Testing...")
        params['train'] = False
        params['load_weights'] = True
        params['episodes'] = 100
        params['batch_size'] = 100
        run_simulation(params)
