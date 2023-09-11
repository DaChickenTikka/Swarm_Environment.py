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


def Real(d):
    c = array([d, ])
    return real(c)


def Imag(p):
    c = array([p, ])
    return imag(c)


def define_parameters():
    params = dict()
    # Neural Network
    params['learning_rate'] = 0.0001
    params['first_layer_size'] = 350  # neurons in the first layer
    params['second_layer_size'] = 35  # neurons in the second layer
    params['third_layer_size'] = 60  # neurons in the third layer
    params['episodes'] = 900
    params['epsilon_decay'] = 0.995
    params['memory_size'] = 100000
    params['batch_size'] = 3000
    # Settings
    params['load_weights'] = False
    params['train'] = True
    params["test"] = False
    params["continue_train"] = False
    params['plot_score'] = True
    ID = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    if params["train"]:
        ID += '_train_'
    if params["test"]:
        ID += '_test_'
    params['weights_path'] = 'weights/20230909130914_train_.h5'
    params['log_path'] = 'logs/scores' + ID + '.txt'
    return params


class Target(Entity):
    def __init__(self, add_to_scene_entities=True, **kwargs):
        super().__init__(add_to_scene_entities, **kwargs)
        self.disabled = 0
        self.respawn = 1
        self.hit_info = self.intersects()

    def update(self):
        self.rotation_y += 100 * time.dt
        self.hit_info = self.intersects()

        if self.hit_info.hit:
            if str(self.hit_info.entity) == "unit":
                self.respawn = 0
                self.hit_info.entities[0].target_count += 1
                self.hit_info.entities[0].success = 1
                #for i in swarm:
                #    if i.position == self.hit_info.entity.position:
                #        i.success = 1
                #        i.movement = False
                #        i.color = color.white

        if (self.x == 0 and self.y == 0) or len(targets) > target_count:
            self.disabled = 1
            self.respawn = 0

        if self.disabled or self.hit_info.hit:
            for i in targets:
                if self.hit_info.hit:
                    i.respawn = 0
                if i.position == self.position:
                    i.disable()
                    targets.remove(i)

        if len(targets) < target_count:  #  and self.respawn:
            targets.append(
                Target(model='cube', texture='vignette',  #scale=(0.75, 0.75, 0.75),
                       collider='box', origin_y=-.5, color=color.violet,
                       x=random.randint(round(-(base_scale - 1) / 2) + 2, round((base_scale - 1) / 2) - 2),
                       y=0.1,
                       z=random.randint(round(-(base_scale - 1) / 2) + 2, round((base_scale - 1) / 2) - 2)))


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
        self.hit_info = 0
        self.vision = []
        self.vision_lines = 7
        self.len_vision = base_scale * 0.5
        self.time_lapsed = 0
        self.final_time = 0
        self.move = [0, 0, 0]  # [turn left, go forward, turn right]
        self.previous_position = self.world_position
        self.stationary = False
        self.stationary_start = time.time()
        self.rewarded = 0
        self.disabled = False
        self.progress = base_scale
        self.reward = 0
        self.respawn = False
        self.target_count = 0

        origin = self.world_position + (
                self.up * 0.5)

        if self.vision_lines % 2 == 0:
            self.vision_lines += 1
        x = 0
        while x < self.vision_lines:
            if self.vision_lines == 1:
                dir = self.direction
            else:  # set the new outer bound
                reference_angle = self.rotation_y * pi / 180 + pi * 3 / 4
                difference = (pi / 2) * (x / (self.vision_lines - 1))
                direction = rect(1, reference_angle - difference)
                dirx = Real(direction)[0]
                dirz = Imag(direction)[0]
                dir = Vec3(-dirx, 0, dirz).normalized()

            self.vision.append(raycast(origin, direction=dir, ignore=(self, Unit, Wall), distance=self.len_vision, debug=True))

            x += 1

    def update(self):
        #  controlling spawn behaviour and size and eliminating swarm if it somehow passes the boundaries
        while len(swarm) < swarm_count and self.respawn:
            swarm.append(Unit(model='cube', collider='box', origin_y=-.5, color=color.blue,
                              x=random.randint(-round((base_scale - 1) / 2 + 3), round((base_scale - 1) / 2) - 3),
                              y=0.1,
                              z=random.randint(-round((base_scale - 1) / 2 + 3), round((base_scale - 1) / 2) - 3),
                              rotation_y=random.randint(0, 360)))

        while len(swarm) > swarm_count:
            swarm[len(swarm) - 1].disable()
            swarm.remove(swarm[len(swarm) - 1])

        if self.x > round((base_scale + 1) / 2) or self.x < -round((base_scale + 1) / 2) or self.z > round(
                (base_scale + 1) / 2) or self.z < -round((base_scale + 1) / 2):
            for i in swarm:
                if i.position == self.position:
                    self.x = random.randint(-round((base_scale - 1) / 2 + 3), round((base_scale - 1) / 2) - 3)
                    self.z = random.randint(-round((base_scale - 1) / 2 + 3), round((base_scale - 1) / 2) - 3)
                    self.disabled = True

        if not self.movement:
            pass
        else:
            # rotation control
            self.rotation_y += (self.move[2] - self.move[0]) * time.dt * 400

            # direction control
            self.direction = (Vec3(sin(self.rotation_y * pi / 180), 0, cos(self.rotation_y * pi / 180)))
            # the ray should start slightly up from the ground, so we can walk up slopes or walk over small objects.
            origin = self.world_position + (
                    self.up * 0.5)

            # hit info based on a ray projected from the object
            self.hit_info = self.intersects()
            self.previous_position = self.position  # save the current position of the cube before it updates

            # check if the cube is stationary
            if self.position == self.previous_position:
                self.stationary = True
            else:
                self.stationary_start = time.time()
                self.stationary = False

            # if the ray doesn't hit anything, allow movement
            if not self.hit_info.hit:
                self.position += self.direction * 5 * time.dt * self.move[1]

            # if the ray hits a target, disable the target, set the cube to have succeeded, then spawn in a new random
            # target
            elif str(self.hit_info.entity) == "target":
                self.success = 1
                #self.movement = False
                #self.color = color.white
                for i in targets:
                    i.respawn = 0
                    if i.position == self.hit_info.entity.position:
                        i.disabled = 1

            # if a success metric has been achieved, do the following:
            if self.success:
                self.target_count += 1
                # for i in swarm:
                #    if i.position == self.position:
                #        i.respawn = False
                #        swarm.remove(i)
                # i.disable()

            # elif str(self.hit_info.entity) == "north" or "south" or "east" or "west" or "wall":
            #    self.wall = True

            # if success hasn't been achieved, then allow the cube to refresh its personal timer and its vision rays
            if not self.success:
                # check if the cube is stationary
                if self.position == self.previous_position:
                    self.stationary = True

                # setting up the vision of the swarm unit
                x = 0
                while x < len(self.vision):
                    if self.vision_lines == 1:
                        dir = self.direction
                    else:
                        reference_angle = self.rotation_y * pi / 180 + pi * 3 / 4
                        difference = (pi / 2) * (x / (self.vision_lines - 1))
                        direction = rect(1, reference_angle - difference)
                        dirx = Real(direction)[0]
                        dirz = Imag(direction)[0]
                        dir = Vec3(-dirx, 0, dirz).normalized()
                    self.vision[x] = raycast(origin, dir, ignore=(self,), distance=self.len_vision, debug=False)
                    x += 1

                for v in self.vision:
                    if not v.hit:
                        self.color = color.blue
                    else:
                        # print(f"{self.position} sees first {x.entity}, total {x.entities}")
                        if str(v.entity) == "target":
                            # print(f"{distance(self, x.entity)}")
                            self.color = color.red
                            break
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
        self.time_lapsed = 0

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
                self.up * 0.1)
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
    fit_reg = False if train == False else True
    ax = sns.regplot(
        x=array([array_counter]),
        y=array([array_score]),
        #color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg=fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [mean(array_score)]*len(array_counter)
    ax.plot(array_counter, y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='Episode', ylabel='Total Reward')
    plt.savefig(fname='plots/'+params['log_path'].lstrip('logs/scores_').rstrip(".txt")+'_plot.png', format='png')


def get_mean_stdev(array):
    return statistics.mean(array), statistics.stdev(array)


def test(params):
    params['load_weights'] = True
    params['train'] = False
    params["test"] = True
    score = run_simulation(params)
    return score


def initialize(unit, list_targets, agent, batch_size):
    state_init1 = agent.get_state(unit, list_targets, walls, swarm_count, target_count, base_scale)  # get the state of the unit
    unit.move = [0, 1, 0]
    state_init2 = agent.get_state(unit, list_targets, walls, swarm_count, target_count, base_scale)
    reward1 = agent.set_reward(unit, list_targets, swarm_count, base_scale, time_limit, done)
    agent.remember(state_init1, unit.move, reward1, state_init2, done)
    agent.replay_new(agent.memory, batch_size)


def run_simulation(params):
    window.vsync = False
    app = Ursina(size=(1800, 900))

    global done
    done = 0

    #  set the size scaling you want for the world
    global base_scale
    base_scale = 40

    # set the time limit in seconds
    # noinspection PyGlobalUndefined
    global time_limit
    time_limit = 10

    # the ground for the swarm to work from
    ground = Entity(model='plane', scale=(base_scale, 1, base_scale), color=color.yellow.tint(-.2),
                    texture='white_cube',
                    texture_scale=(base_scale, base_scale), collider='box')
    ground.eternal = True

    timer = Text(text="", x=-0.98, y=0.44)  # setting up a timer to record and display the speed of the units
    Target_count = Text(text='', x=-0.98, y=0.4)
    swarm_counter = Text(text='', x=-0.98, y=0.36)
    episode_count = Text(text='', x=-0.98, y=0.48)
    train_test = Text(text='', x=0.90, y=0.44)
    timer.eternal = True
    Target_count.eternal = True
    swarm_counter.eternal = True
    episode_count.eternal = True
    train_test.eternal = True

    global walls
    walls = []

    # set the number of targets and/or swarm units you want
    global target_count
    target_count = 8
    global swarm_count
    swarm_count = 4

    # the boundaries of the world to contain the units on the ground
    #north = Wall(name='North', model='cube', collider='box', scale=(base_scale, base_scale, 2), color=color.white10)
    #south = Wall(name='South', model='cube', collider='box', scale=(base_scale, base_scale, 2), color=color.white10)
    #east = Wall(name='East', model='cube', collider='box', scale=(base_scale, base_scale, 2), rotation=(0, 90, 0),
    #            color=color.white10)
    #west = Wall(name='West', model='cube', collider='box', scale=(base_scale, base_scale, 2), rotation=(0, 90, 0),
    #            color=color.white10)
    #north.z = base_scale / 2 + 1
    #north.eternal = True
    #south.z = -base_scale / 2 - 1
    #south.eternal = True
    #east.x = base_scale / 2 + 1
    #east.eternal = True
    #west.x = -base_scale / 2 - 1
    #west.eternal = True

    wall_generate = 0

    while wall_generate < base_scale:
        walls.append(Wall(model='cube', collider='box', x=base_scale / 2 - wall_generate - 0.5, z=base_scale / 2 + 0.5,
                          color=color.white10))
        walls.append(Wall(model='cube', collider='box', x=base_scale / 2 - wall_generate - 0.5, z=-base_scale / 2 - 0.5,
                          color=color.white10))
        walls.append(Wall(model='cube', collider='box', x=base_scale / 2 + 0.5, z=base_scale / 2 - wall_generate - 0.5,
                          color=color.white10))
        walls.append(Wall(model='cube', collider='box', x=-base_scale / 2 - 0.5, z=base_scale / 2 - wall_generate - 0.5,
                          color=color.white10))
        wall_generate += 1

    for i in walls:
        i.eternal = True



    # initialise the camera to view the world
    viewer = Camera()
    viewer.eternal = True
    viewer.y = 15
    viewer.z = -25

    agent = SwarmTargetDQNAgent(params)
    agent = agent.to(DEVICE)
    agent.optimizer = optim.Adam(agent.parameters(), weight_decay=0, lr=params['learning_rate'])

    global counter_episodes
    if params['continue_train']:
        counter_episodes = 400
    else:
        counter_episodes = 0
    reward_plot = []
    time_plot = []

    define_parameters()

    app.step()
    app.step()
    app.step()

    while counter_episodes <= params['episodes']:
        # resetting the targets and swarm units for a new round of training
        scene.clear()
        total_reward = 0
        started = False

        # initialise the lists that will contain the swarm and target objects
        global swarm
        swarm = []
        global targets
        targets = []

        if params['train'] or params['continue_train']:
            train_test.text = 'Training'

        if params['test']:
            train_test.text = 'Testing'

        # walls_count = 25

        # place the targets and swarm units in random locations across the worlds
        i = 0
        while i < swarm_count:
            swarm.append(Unit(model='cube', collider='box', origin_y=-.5, color=color.blue,
                              x=random.randint(-round((base_scale - 1) / 2 - 3), round((base_scale - 1) / 2) - 3),
                              y=0.1,
                              z=random.randint(-round((base_scale - 1) / 2 - 3), round((base_scale - 1) / 2) - 3),
                              rotation_y=random.randint(0, 360)))
            i += 1

        if params['train']:
            pass
            # target_count = int(ceil(5*(pow(0.9972, counter_episodes))))
        for i in range(target_count + 2):
            targets.append(
                Target(model='cube', texture='vignette',  # scale=(0.75, 0.75, 0.75),
                       collider='box', origin_y=-.5,
                       color=color.violet, x=random.randint(round(-(base_scale - 1) / 2) + 2, round((base_scale - 1) / 2) - 2),
                       y=0.1, z=random.randint(round(-(base_scale - 1) / 2) + 2, round((base_scale - 1) / 2) - 2)))

        global start_time
        start_time = time.time()

        app.step()
        app.step()
        app.step()

        for i in swarm:
            initialize(i, targets, agent, params['batch_size'])
        #  wall_left = Entity(model='cube', collider='box', scale_y=3, scale_x = 6, origin_y=-.5, color=color.azure, x=-4)
        app.step()

        if not params['train']:
            agent.epsilon = 0.02
        else:
            # agent.epsilon is set to give randomness to actions
            agent.epsilon = power(params["epsilon_decay"], counter_episodes)
            if agent.epsilon < 0.02:
                agent.epsilon = 0.02

        print(f"Start of episode {counter_episodes}")
        done = 0
        time_lapsed = 0

        while time_lapsed < time_limit and len(swarm) > 0:

            # set the start time at the start of the time loop once, then don't update again until next episode
            if not started:
                # set the start time as the beginning of the game
                start_time = time.time()
                started = True

            total_targets = 0
            for t in swarm:
                total_targets += t.target_count

            time_lapsed = time.time() - start_time  # subtract the current time from the start time for the time elapsed
            timer.text = "Time: " + str(round(time_lapsed, 3)) + 's'  # print the timer
            swarm_counter.text = "Units Remaining: " + str(len(swarm))  # print the target counter
            episode_count.text = "Episode " + str(counter_episodes)     # print the episode counter
            Target_count.text = "Targets Found: " + str(total_targets)  # print the target counter

            # get old state
            for i in swarm:
                # updating the time information for use in the machine learning in the individual units
                i.time_lapsed = time_lapsed
                state_old = agent.get_state(i, targets, walls, swarm_count, target_count, base_scale)

                # perform random actions based on agent.epsilon, or choose the action
                i.choice = random.uniform(0, 1)
                if i.choice < agent.epsilon:
                    final_move = eye(3)[randint(0, 2)]
                else:
                    # predict action based on the old state
                    with torch.no_grad():
                        state_old_tensor = torch.tensor(state_old.reshape((1, 9)), dtype=torch.float32).to(
                            DEVICE)
                        prediction = agent(state_old_tensor)
                        final_move = eye(3)[argmax(prediction.detach().cpu().numpy()[0])]
                    # perform new move and get new state
                i.move[0] = final_move[0]
                i.move[1] = final_move[1]
                i.move[2] = final_move[2]
            app.step()
            for i in swarm:
                i.time_lapsed = time_lapsed
                state_new = agent.get_state(i, targets, walls, swarm_count, target_count, base_scale)
                # set reward for the new state
                Reward = agent.set_reward(i, targets, swarm_count, base_scale, time_limit, done)
                i.reward = Reward
                # print(f"unit: ({swarm.index(i)}), move: {i.move}, reward: {i.reward}, epsilon: {agent.epsilon}, choice: {i.choice}")
                total_reward += Reward

                if params['train'] or params['continue_train']:
                    # train short memory base on the new action and state
                    agent.train_short_memory(state_old, final_move, Reward, state_new, done)
                    # store the new data into a long term memory
                    agent.remember(state_old, final_move, Reward, state_new, done)

                    model_weights = agent.state_dict()
                    torch.save(model_weights, params["weights_path"])

            if time.time() - start_time > time_limit or len(targets) == 0 or len(swarm) == 0:
                break

        done = 1
        print("####################")
        print("   End of Episode")
        print("####################")
        for i in swarm:
            state_new = agent.get_state(i, targets, walls, swarm_count, target_count, base_scale)
            Reward = agent.set_reward(i, targets, swarm_count, base_scale, time_limit, done)
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

        print(f"Total Episode Reward: ({total_reward})")
        if params['train'] or params['continue_train']:
            agent.replay_new(agent.memory, params['batch_size'])
        reward_plot.append(total_reward)
        time_plot.append(counter_episodes)
        counter_episodes += 1
        if params['plot_score'] and counter_episodes % 5 == 0:
            plot_seaborn(time_plot, reward_plot, params['train'])
    if params['plot_score']:
        plot_seaborn(time_plot, reward_plot, params['train'])
    mean, stdev = get_mean_stdev(reward_plot)

    return total_reward, mean, stdev


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    params = define_parameters()
    parser.add_argument("--display", nargs='?', type=distutils.util.strtobool, default=True)
    parser.add_argument("--bayesianopt", nargs='?', type=distutils.util.strtobool, default=False)#not params['test'])
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
