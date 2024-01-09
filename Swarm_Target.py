#  27th May 2023, design decision made, blocks can rotate on the spot and only move forward, happy birthday to myself I guess XD
import datetime
import statistics
from cmath import rect
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
from ursina import *
from DQN import DDQNAgent

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Using " + DEVICE)


def Real(d):
    c = np.array([d, ])
    return np.real(c)


def Imag(p):
    c = np.array([p, ])
    return np.imag(c)


def define_parameters():
    paramse = dict()
    # Spectator Settings
    paramse["view_mode"] = "F"  # [F]ull view, [P]artial view
    # Simulation Settings
    paramse["time_limit"] = 15
    paramse["base_scale"] = 20
    paramse["swarm_size"] = 1
    paramse["targets_amount"] = 3
    paramse['episodes'] = 150
    paramse['train'] = False
    paramse["test"] = False
    paramse["continue_train"] = False
    paramse['plot_score'] = True
    # Neural Network
    paramse['first_layer_size'] = 64  # neurons in the first layer
    paramse['second_layer_size'] = 96  # neurons in the second layer
    paramse['third_layer_size'] = 64  # neurons in the third layer
    paramse['learning_rate'] = 3e-4  # 0.01
    paramse['gamma'] = 0.999
    paramse['tau'] = 1e-3
    paramse['memory_size'] = int(1e5)
    paramse['batch_size'] = 128
    paramse['episode_count'] = 0
    paramse['epsilon'] = 1  # 0.02 ** ((1.8 * paramse['episode_count']) / paramse['episodes'])
    paramse['time_lapsed'] = 0
    paramse['load_weights'] = False

    return paramse


class Target(Entity):
    def __init__(self, add_to_scene_entities=True, **kwargs):
        super().__init__(add_to_scene_entities, **kwargs)
        self.disabled = 0
        self.hit_info = self.intersects()
        self.name = 0

    def update(self):
        # self.rotation_y += 100 * time.dt
        self.hit_info = self.intersects()
        base_scale = params["base_scale"]

        if self.hit_info.hit:
            if str(self.hit_info.entity) in swarm:
                self.hit_info.entities[0].target_count += 1
                self.hit_info.entities[0].success = 1

        if (self.x == 0 and self.y == 0) or len(targets) > params["targets_amount"]:
            self.disabled = 1

        if self.disabled or (str(self.hit_info.entity)[0] == "S") or (len(targets) < params["targets_amount"]):
            while 1:
                safe = True
                sx = random.randint(round(-(base_scale - 1) / 2) + 4, round((base_scale - 1) / 2) - 4)
                sz = random.randint(round(-(base_scale - 1) / 2) + 4, round((base_scale - 1) / 2) - 4)

                scale = 2 * ceil(np.log(base_scale))
                for a in swarm:
                    if swarm[a].x + scale > sx > swarm[a].x - scale and swarm[a].z + scale > sz > swarm[a].z - scale:
                        safe = False
                #for a in targets:
                #    if targets[a].x + scale > sx > targets[a].x - scale and targets[a].z + scale > sz > targets[
                #        a].z - scale:
                #        safe = False
                if safe:
                    break
            self.x = sx
            self.y = 0.1
            self.z = sz
            self.disabled = 0

            if len(targets) < params["targets_amount"]:
                targets["T" + str(len(targets))] = Target(name="T" + str(len(targets)), model='sphere',
                                                          texture='vignette',
                                                          # scale=(0.75, 0.75, 0.75),
                                                          collider='box', origin_y=-.5, color=color.violet,
                                                          x=sx,
                                                          y=0.1,
                                                          z=sz)
                targets["T" + str(len(targets) - 1)].name = "T" + str(len(targets) - 1)


class Wall(Entity):
    def __init__(self, add_to_scene_entities=True, **kwargs):
        super().__init__(add_to_scene_entities, **kwargs)


class Unit(Entity):

    def __init__(self, add_to_scene_entities=True, **kwargs):
        super().__init__(add_to_scene_entities, **kwargs)
        self.rotation_y = 0
        self.direction = (Vec3(sin(self.rotation_y * np.pi / 180), 0, cos(self.rotation_y * np.pi / 180)))
        self.position = self.world_position
        self.movement = True
        self.success = False
        self.vision = []
        self.vision_lines = 7
        self.len_vision = params["base_scale"] * 0.35
        self.move = [0, 0, 0]  # [turn left, go forward, turn right]
        self.choice = 0
        self.state_old = 0
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

        origin = self.world_position + (self.up * 0.5)
        boundary = ceil(2 * np.log10(params["base_scale"]))
        self.detect = boxcast(origin=origin + self.back * boundary / 2, direction=self.direction,
                              thickness=(boundary, 1), distance=boundary, ignore=(self,))
        self.hit_info = self.intersects(ignore=(self.detect,))

        if self.vision_lines % 2 == 0:
            self.vision_lines += 1
        q = 0
        while q < self.vision_lines:
            if self.vision_lines == 1:
                dir = self.direction
            else:  # set the new outer bound
                reference_angle = self.rotation_y * np.pi / 180 + np.pi * 3 / 4
                difference = (np.pi / 2) * (q / (self.vision_lines - 1))
                direction = rect(1, reference_angle - difference)
                dirx = Real(direction)[0]
                dirz = Imag(direction)[0]
                dir = Vec3(-dirx, 0, dirz).normalized()

            self.vision.append(
                raycast(origin, direction=dir, ignore=(self, Unit, Wall, self.detect), distance=self.len_vision,
                        debug=False))
            q += 1

    def update(self):
        base_scale = params["base_scale"]
        #  controlling spawn behaviour and size and eliminating swarm if it somehow passes the boundaries

        if self.x > round((base_scale + 1) / 2) - 0.35 or self.x < -round((base_scale + 1) / 2) + 0.35 or \
                self.z > round((base_scale + 1) / 2) - 0.35 or self.z < -round((base_scale + 1) / 2) + 0.35:
            self.disabled = 1
            while 1:
                safe = True
                sx = random.randint(round(- base_scale / 2) + 3, round(base_scale / 2) - 3)
                sz = random.randint(round(- base_scale / 2) + 3, round(base_scale / 2) - 3)
                for a in targets:
                    if targets[a].x + 2 > sx > targets[a].x - 2 and targets[a].z + 2 > sz > targets[a].z - 2:
                        safe = False
                    size_scale = 2.2 * (base_scale ** (2 * params['episode_count'] / params['episodes']))
                    if sqrt((sx - targets[a].x) ** 2 + (sz - targets[a].z) ** 2) > size_scale:
                        safe = False
                if safe:
                    break
            self.x = sx
            self.z = sz
            self.look_at(target=targets["T0"])
            temp_rotation = self.rotation_y
            self.rotation_y = temp_rotation + 45 * (
                float(round((params['episode_count'] / params['episodes']) ** 4, 2))) * (
                                      (-1) ** params['episode_count'])

        # encouragement for the unit to explore more if it's not able to see a target
        if params['train'] and distance(self, targets[str(max(targets))]) > 0:  # self.len_vision/10:
            dist = distance(self, targets[str(max(targets))])

            # reward unit for approaching target
            if dist > self.max_distance:
                self.max_distance = dist

            if dist <= 5.5 * self.max_distance / 6:
                self.goal_found = 1
                self.max_distance = 0

            # punish unit for leaving target
            if dist < self.prev_min:
                self.prev_min = dist

            if dist ** 2 >= (5.5 / 5) * (
                    self.prev_min ** 2):  # using exponents to increase punishment frequency the further distance away is found
                self.goal_lost = 1
                self.prev_min = dist

        if not self.movement:
            pass
        else:
            self.move = [1 if self.choice == 0 else 0, 1 if self.choice == 1 else 0, 1 if self.choice == 2 else 0]
            # rotation control
            self.rotation_y += (self.move[2] - self.move[0]) * time.dt * 220

            # direction control
            self.direction = (Vec3(sin(self.rotation_y * np.pi / 180), 0, cos(self.rotation_y * np.pi / 180)))
            # the ray should start slightly up from the ground, so we can walk up slopes or walk over small objects.
            origin = self.world_position + (self.up * 0.5)
            boundary = ceil(2 * np.log10(params["base_scale"]))
            self.detect = boxcast(origin=origin + self.back * boundary / 2, direction=self.direction,
                                  thickness=(boundary, 1), distance=boundary, ignore=(self,))

            # hit info based on intersection
            self.hit_info = self.intersects(ignore=(self.detect,))
            # print("Current position: " + str(self.world_position) + ", detect position: " + str(self.detect.world_position) + "Detected entities: " + str(self.detect_entity.entities))
            self.previous_position = self.position  # save the current position of the cube before it updates

            # if the ray doesn't hit anything, allow movement
            if str(self.hit_info.entity) not in swarm and "W" not in str(self.hit_info.entity):
                self.position += self.direction * params["base_scale"] / 4 * time.dt * self.move[1]

            # cube collision
            match str(self.hit_info.entity)[0]:
                # if the unit hits a target, immediately teleport the target, set the cube to have succeeded, then
                # bring the target into a new random location
                case "T":
                    self.success = 1
                    # self.movement = False
                    # self.color = color.white
                    targets[self.hit_info.entity.name].respawn = 0
                    targets[self.hit_info.entity.name].disabled = 1
                    targets[self.hit_info.entity.name].y = -4

                # if the unit hits a wall or another unit, put it back in its original position to prevent
                # phasing into the walls or other units
                case "W":
                    self.position = self.previous_position
                    if self.move[1]:
                        self.position -= self.direction * params["base_scale"] / 4 * time.dt
                    else:
                        self.position -= Vec3(self.hit_info.entity.x - self.x, 0, self.hit_info.entity.z - self.z) * \
                                         params["base_scale"] / 6 * time.dt

            # check if the cube is stationary
            if self.position == self.previous_position:
                self.stationary = True
            # self.move_start = time.time()
            else:
                self.stationary_start = time.time()
                self.stationary = False
            # if a success metric has been achieved, do the following:
            # if self.success:
            #     self.target_count += 1

            # if success hasn't been achieved, then allow the cube to refresh its vision rays
            if not self.success:
                # setting up the vision of the swarm unit
                x = 0
                while x < self.vision_lines:
                    if self.vision_lines == 1:
                        dir = self.direction
                    else:
                        reference_angle = self.rotation_y * np.pi / 180 + np.pi * 3 / 4
                        difference = (np.pi / 2) * (x / (self.vision_lines - 1))
                        direction = rect(1, reference_angle - difference)
                        dirx = Real(direction)[0]
                        dirz = Imag(direction)[0]
                        dir = Vec3(-dirx, 0, dirz).normalized()
                    self.vision[x] = raycast(origin, dir, ignore=(self, self.detect), distance=self.len_vision,
                                             debug=False)
                    x += 1

                for v in self.vision:
                    if not v.hit:
                        self.color = color.blue
                        if self.detect.hit:
                            if str(self.detect.entities[0])[0] == "W":
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
                            if self.detect.hit:
                                if str(self.detect.entities[0])[0] == "W":
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


def plot_seaborn(array_counter, array_score):
    sns.set(color_codes=True, font_scale=1.5)
    sns.set_style("white")
    plt.figure(figsize=(13, 8))
    fit_reg = True
    ax = sns.regplot(
        x=np.array([array_counter]),
        y=np.array([array_score]),
        # color="#36688D",
        x_jitter=.1,
        scatter_kws={"color": "#36688D"},
        label='Data',
        fit_reg=fit_reg,
        line_kws={"color": "#F49F05"}
    )
    # Plot the average line
    y_mean = [np.mean(array_score)] * len(array_counter)
    ax.plot(array_counter, y_mean, label='Mean', linestyle='--')
    ax.legend(loc='upper right')
    ax.set(xlabel='Episode', ylabel='Total Reward')
    plt.savefig(fname=params['plot_path'], format='png')


def get_mean_stdev(arr):
    return statistics.mean(arr), statistics.stdev(arr)


def test(paramss, weightspath):
    paramss['load_weights'] = True
    paramss['plot_score'] = True
    paramss['weights_path'] = 'weights/' + weightspath + '.h5'
    paramss['plot_path'] = 'plots/' + weightspath + '_plot.png'
    paramss['train'] = False
    paramss['continue_train'] = False
    paramss["test"] = True
    paramss['time_limit'] = 50
    paramss['episode_count'] = 1500
    paramss['episodes'] = 1530
    print("Testing...")
    score = run_simulation(paramss)
    return score


def train(paramss):
    paramss['load_weights'] = False
    paramss['train'] = True
    ID = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    ID += '_train_'
    paramss['weights_path'] = 'weights/' + ID + '.h5'
    paramss['plot_path'] = 'plots/' + ID + '_plot.png'
    paramss['continue_train'] = False
    paramss["test"] = False
    paramss['episode_count'] = 0
    paramss['episodes'] = 150
    print("Training...")
    score = run_simulation(paramss)
    return score


def continuetrain(paramss, weightspath):
    paramss['load_weights'] = True
    paramss['weights_path'] = 'weights/' + weightspath + '_cont.h5'
    paramss['plot_path'] = 'plots/' + weightspath + '_cont_plot.png'
    paramss['train'] = False
    paramss['continue_train'] = True
    paramss["test"] = False
    paramss['time_limit'] *= 2
    paramss['episodes'] = 1500
    paramss['episode_count'] = int(paramss['episodes'] / 2)
    print("Continuing Training...")
    score = run_simulation(paramss)
    return score


def set_reward(unit):
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
    reward = 0
    if unit.success:
        reward = 1
        print("###################################################")
        print(f"{unit.name} succeeded and rewarded, total targets {unit.target_count}")
        print("###################################################")
        unit.success = 0
        return reward

    if not unit.success:

        # punishing stationary units, contact with other units and walls, rewarding movement slightly
        if not unit.stationary:
            reward = 0  # 0.001  #(time.time() - unit.stationary_start)*2
        else:
            reward = -0.0005

        if unit.detect.hit:
            if str(unit.detect.entities[0])[0] == "U":
                reward -= 0.0001
            if str(unit.detect.entities[0])[0] == "W":
                reward -= 0.0001
        if unit.disabled == 1:
            unit.disabled = 0
            print("########### Unit Disabled ###########")

    return round(reward, 4)


def get_state(unit):
    """
    - Plan, reduce the vision down to a few neurons of "Unit seen, wall seen, or target seen, all broken down to a left, right and center vision"
    Return the state.
    The state is a numpy array of 7 values, representing:
        - 0, 1, 2: Unit sees target on its [left, center, right]
        - 3: Unit is close to a wall
        - 4, 5, 6: Unit sees wall on its [left, center, right]
    """

    vision_range = unit.vision_lines
    left_target = False
    left_wall = False
    left_unit = False
    left_unit_red = False
    right_target = False
    right_wall = False
    right_unit = False
    right_unit_red = False
    center_target = False
    center_wall = False
    center_unit = False
    center_unit_red = False
    wall_detect = False

    for i in unit.vision:
        if unit.vision.index(i) < int((vision_range - 1) / 2):
            if i.hit:
                for l in i.entities:
                    if str(l)[0] == "T":
                        right_target = True
                    if str(l)[0] == "W":
                        right_wall = True

        if unit.vision.index(i) > int((vision_range - 1) / 2):
            if i.hit:
                for l in i.entities:
                    if str(l)[0] == "T":
                        left_target = True
                    if str(l)[0] == "W":
                        left_wall = True

        if unit.vision.index(i) == int((vision_range - 1) / 2):
            if i.hit:
                for l in i.entities:
                    if str(l)[0] == "T":
                        center_target = True
                    if str(l)[0] == "W":
                        center_wall = True

    if unit.detect.hit:
        if str(unit.detect.entities[0])[0] == "W":
            wall_detect = True

    state = [
        left_target,

        center_target,

        right_target,

        wall_detect,

        left_wall,

        center_wall,

        right_wall
    ]

    for i in range(len(state)):
        if state[i]:
            state[i] = 1
        else:
            state[i] = 0

    return np.asarray(state)


def run_simulation(params):
    print("Initialising Simulation...")

    match params["view_mode"]:

        # Full View
        case "F":
            app = Ursina(size=(1800, 900))
            timer = Text(text="", x=-0.98,
                         y=0.44)  # setting up a timer to record and display the time of the simulation
            Target_count = Text(text='', x=-0.98, y=0.4)
            swarm_counter = Text(text='', x=-0.98, y=0.36)
            episode_count = Text(text='', x=-0.98, y=0.48)

        # Partial View
        case "P":
            app = Ursina(size=(1000, 650), position=(920, 0))
            timer = Text(text="", x=-0.73,
                         y=0.44)  # setting up a timer to record and display the time of the simulation
            Target_count = Text(text='', x=-0.73, y=0.40)
            swarm_counter = Text(text='', x=-0.73, y=0.36)
            episode_count = Text(text='', x=-0.73, y=0.48)

        # default case, Full View
        case _:
            app = Ursina(size=(1800, 900))
            timer = Text(text="", x=-0.98,
                         y=0.44)  # setting up a timer to record and display the time of the simulation
            Target_count = Text(text='', x=-0.98, y=0.4)
            swarm_counter = Text(text='', x=-0.98, y=0.36)
            episode_count = Text(text='', x=-0.98, y=0.48)

    window.exit_button.enabled = False
    window.vsync = False  # set the size scaling you want for the world
    base_scale = params["base_scale"]

    print("Setting up world...")
    # the ground for the swarm to work from
    ground = Entity(model='plane', scale=(base_scale, 1, base_scale), color=color.yellow.tint(-.2),
                    texture='white_cube',
                    texture_scale=(base_scale, base_scale), collider='box')
    ground.eternal = True

    timer.eternal = True
    Target_count.eternal = True
    swarm_counter.eternal = True
    episode_count.eternal = True

    walls = {}

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

    ################################################################################################################################################################################################################################################
    #   Agent Preparation - beginning of training
    ################################################################################################################################################################################################################################################
    print("Preparing DQN...")
    ddqnagent = DDQNAgent(7, 3, 0, params)
    if params['test'] or params['continue_train']:
        ddqnagent.qnetwork_local.load_state_dict(
            torch.load(params['weights_path'], map_location=lambda storage, loc: storage))
        print("Loaded DQN")

    print("DQN prepared, beginning episodes...")
    reward_plot = []
    time_plot = []

    ################################################################################################################################################################################################################################################
    #  Start of Episode
    ################################################################################################################################################################################################################################################
    goal_reached = 0
    while (params['episode_count'] <= params['episodes']) or ((not goal_reached and params['train']) or params['test']):
        # resetting the targets and swarm units for a new round of training
        scene.clear()
        total_reward = 0

        # initialise the lists that will contain the swarm and target objects
        global swarm
        swarm = {}
        global targets
        targets = {}

        ################################################################################################################################################################################################################################################
        #   Spawning units, targets and internal walls
        ################################################################################################################################################################################################################################################

        # place the targets and swarm units in random locations across the worlds
        j = len(targets)
        while j < params["targets_amount"]:
            targets["T" + str(j)] = Target(name="T" + str(j), model='cube', texture='vignette',
                                           # scale=(0.75, 0.75, 0.75),
                                           collider='box', origin_y=-.5,
                                           color=color.violet,
                                           x=random.randint(round(-(base_scale - 1) / 2) + 4,
                                                            round((base_scale - 1) / 2) - 4),
                                           y=0.1,
                                           z=random.randint(round(-(base_scale - 1) / 2) + 4,
                                                            round((base_scale - 1) / 2) - 4))
            targets["T" + str(j)].name = "T" + str(j)
            targets["T" + str(j)].targets_found = 0
            j = len(targets)

        i = len(swarm)
        while i < params["swarm_size"]:
            while 1:
                safe = True
                sx = random.randint(round(-base_scale / 2) + 3, round((base_scale / 2) - 3))
                sz = random.randint(round(-base_scale / 2) + 3, round((base_scale / 2) - 3))
                for a in targets:
                    if targets[a].x + 2 > sx > targets[a].x - 2 and targets[a].z + 2 > sz > targets[a].z - 2:
                        safe = False
                for b in swarm:
                    if swarm[b].x + 2 > sx > swarm[b].x - 2 and swarm[b].z + 2 > sz > swarm[b].z - 2:
                        safe = False
                    size_scale = (base_scale ** (4 * params['episode_count'] / params['episodes']))
                    if sqrt((sx - targets[a].x) ** 2 + (sz - targets[a].z) ** 2) > size_scale:
                        safe = False
                if safe:
                    break
            swarm["U" + str(i)] = Unit(name="U" + str(i), model='cube', collider='box', origin_y=-.5, color=color.blue,
                                       x=sx, y=0.1, z=sz)
            swarm["U" + str(i)].name = "U" + str(i)
            swarm["U" + str(i)].look_at(target=targets["T0"])
            temp_rotation = swarm["U" + str(i)].rotation_y
            diff = float(round((params['episode_count'] / params['episodes']) ** 4, 2))
            if diff > 1:
                diff = 1
            swarm["U" + str(i)].rotation_y = temp_rotation + 45 * diff * ((-1) ** params['episode_count'])

            i = len(swarm)

        ################################################################################################################################################################################################################################################
        #   Beginning of timer
        ################################################################################################################################################################################################################################################

        global start_time
        start_time = time.time()

        if not params['train']:
            ddqnagent.epsilon = 0.01
        else:
            # agent.epsilon is set to give randomness to actions
            if params["epsilon"] < 0.02:
                ddqnagent.epsilon = 0.02
            else:
                ddqnagent.epsilon = params["epsilon"]

        print(f"Start of episode {params['episode_count']}, epsilon : {params['epsilon']}")
        done = 0
        started = False
        params['time_lapsed'] = 0
        app.step()
        while not done:
            # set the start time at the start of the time loop once, then don't update again until next episode
            if not started:
                # set the start time as the beginning of the episode
                start_time = time.time()
                started = True
            else:
                total_targets = 0
                for t in swarm:
                    total_targets += swarm[str(t)].target_count

                params[
                    'time_lapsed'] = time.time() - start_time  # subtract the current time from the start time for the time elapsed
                timer.text = "Time: " + str(round(params['time_lapsed'], 3)) + 's'  # print the timer
                swarm_counter.text = "Units Remaining: " + str(len(swarm))  # print the target counter
                episode_count.text = "Episode " + str(params['episode_count'])  # print the episode counter
                Target_count.text = "Targets Found: " + str(int(total_targets))  # print the target counter
                # move units based on current state
                for i in swarm:
                    # get current state
                    swarm[i].state_old = get_state(swarm[i])
                    # perform random actions based on agent.epsilon, or choose the action
                    swarm[i].choice = ddqnagent.act(swarm[i].state_old, ddqnagent.epsilon)
                if held_keys['escape'] and params['plot_score']:
                    plot_seaborn(time_plot, reward_plot)
                    quit()
                app.step()

                update = False
                for i in swarm:
                    if swarm[i].success:
                        update = True
                        break
                if update and params['train']:
                    params['time_limit'] += 3

                # check if an end condition has been reached
                if (params['time_lapsed'] >= params['time_limit']) or (total_targets > 25):
                    done = 1

                # reward agent based on move and update model weights
                for i in swarm:
                    # get new state after move
                    state_new = get_state(swarm[i])
                    # set reward for the new state
                    Reward = set_reward(swarm[i])
                    total_reward += Reward
                    if not params['test']:
                        ddqnagent.step(swarm[i].state_old, swarm[i].choice, Reward, state_new, done, params)
                        # print(f"unit: {i}, hit: {swarm[i].hit_info.entity}, detect: {swarm[i].detect_entity.entity}, move: {swarm[i].move}, reward: {swarm[i].reward}, epsilon: {agent.epsilon}, choice: {swarm[i].choice}")
                        model_weights = ddqnagent.qnetwork_local.state_dict()
                        torch.save(model_weights, params["weights_path"])

        print("####################")
        print("   End of Episode   ")
        print("####################")

        reward_plot.append(total_reward)

        # if the AI finds and captures 10 targets (reward will be less than 10 in this scenario), check if it has been successfully trained
        if (len(reward_plot) > 15) and (total_targets > 9) and params['train']:
            goal_reached = 1
            for i in range(
                    10):  # for a successfully trained AI, the unit has to have gotten more than 3 targets in the 20 rounds prior
                if reward_plot[params['episode_count'] - i] < 4:
                    goal_reached = 0
                    break

        params['time_limit'] -= 3 * total_targets
        time_plot.append(params['episode_count'])
        print(f"Total Episode Reward: ({total_reward})")
        params['episode_count'] += 1
        if params['train']:
            params['epsilon'] = 0.02 ** ((1.8 * params['episode_count']) / params['episodes'])
        scene.clear()
        app.step()

    if params['plot_score']:
        plot_seaborn(time_plot, reward_plot)
    # meaN, stdev = get_mean_stdev(reward_plot)

    # clear scene after simulation run
    ground.eternal = False
    timer.eternal = False
    Target_count.eternal = False
    swarm_counter.eternal = False
    episode_count.eternal = False
    for i in walls:
        walls[i].eternal = False
    viewer.eternal = False
    scene.clear()

    # return total_reward, meaN, stdev
    return 0


if __name__ == '__main__':
    params = define_parameters()
    train(params)
    test(params, params['weights_path'].lstrip("weights/").rstrip(".h5"))
