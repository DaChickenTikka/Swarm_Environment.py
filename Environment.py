#  27th May 2023, design decision made, blocks can rotate on the spot and only move forward, happy birthday to myself I guess XD

from ursina import *
from cmath import rect
from numpy import real, imag, pi, array, sin, cos

if __name__ == '__main__':

    window.vsync = False
    app = Ursina(size=(1800, 900))

    def Real(x):
        c = array([x, ])
        return real(c)

    def Imag(x):
        c = array([x, ])
        return imag(c)

    # the ground for the swarm to work from
    base_scale = 101
    ground = Entity(model='plane', scale=(base_scale, 1, base_scale), color=color.yellow.tint(-.2), texture='white_cube',
                    texture_scale=(base_scale, base_scale), collider='box')

    swarm = []
    targets = []

    class Target(Entity):
        def __init__(self, add_to_scene_entities=True, **kwargs):
            super().__init__(add_to_scene_entities, **kwargs)
            self.touched = False
            self.hit_info = boxcast(self.world_position - (0.5, 0, 0.5), (0, 0, 1), 1, (1, 1),  ignore=(self, ), debug=False)

        def update(self):
            if not self.touched:
                self.rotation_y += 100 * time.dt
                self.hit_info = raycast(self.world_position, ignore=(self, ), distance=0.01, debug=False)
                if self.touched:
                    self.disable()

    class Wall(Entity):
        def __init__(self, add_to_scene_entities=True, **kwargs):
            super().__init__(add_to_scene_entities, **kwargs)

    class Unit(Entity):
        def __init__(self, add_to_scene_entities=True, **kwargs):
            super().__init__(add_to_scene_entities, **kwargs)
            self.rotation_y = 0
            self.direction = (Vec3(sin(self.rotation_y * pi/180), 0, cos(self.rotation_y * pi/180)))
            self.position = self.world_position
            self.movement = True
            self.success = False
            self.hit_info = 0
            self.vision = []
            self.vision_lines = 7
            self.state = [self.position, self.rotation]

            origin = self.world_position + (
                    self.up * 0.5)

            if self.vision_lines % 2 == 0:
                self.vision_lines += 1
            x = 0
            while x < self.vision_lines:
                if self.vision_lines == 1:
                    dir = self.direction
                else:       ## set the new outer bound
                    reference_angle = self.rotation_y * pi/180 + pi*3/4
                    difference = (pi/2) * (x / (self.vision_lines - 1))
                    direction = rect(1, reference_angle - difference)
                    dirx = Real(direction)[0]
                    dirz = Imag(direction)[0]
                    dir = Vec3(-dirx, 0, dirz).normalized()

                self.vision.append(raycast(origin, direction=dir, ignore=(self, Unit, Wall), distance=40, debug=True))

                x += 1

        def update(self):
            # if the cube is allowed to move
            if self.movement:
                # rotation control
                self.rotation_y += (held_keys['k'] - held_keys['h']) * time.dt * 10 * random.randint(0, 25) * .5

                # direction control
                self.direction = (Vec3(sin(self.rotation_y * pi/180), 0, cos(self.rotation_y * pi/180)))
                # the ray should start slightly up from the ground, so we can walk up slopes or walk over small objects.
                origin = self.world_position + (
                            self.up * 0.5)

                # hit info based on a ray projected from the object, like light
                self.hit_info = raycast(origin, self.direction, ignore=(self, Unit, ), distance=.5, debug=False)
                # if the ray doesn't hit anything, keep moving
                if not self.hit_info.hit:
                    self.position += self.direction * 5 * time.dt * (held_keys['u'] - held_keys['j']) * random.randint(3, 25) * .1

                # if a success metric has been achieved, stop moving the cube and set the color to white
                if self.success:
                    self.movement = False
                    self.color = color.white
                # if not, then allow the cube to refresh its vision rays
                else:
                    x = 0
                    while x < len(self.vision):
                        if self.vision_lines == 1:
                            dir = self.direction
                        else:
                            reference_angle = self.rotation_y * pi / 180 + pi * 3 / 4
                            difference = (pi/2) * (x / (self.vision_lines - 1))
                            direction = rect(1, reference_angle - difference)
                            dirx = Real(direction)[0]
                            dirz = Imag(direction)[0]
                            dir = Vec3(-dirx, 0, dirz).normalized()
                        self.vision[x] = raycast(origin, dir, ignore=(self, Unit, Wall), distance=40, debug=False)
                        x += 1

                    for x in self.vision:
                        if not x.hit:
                            self.color = color.blue
                        else:
                            print(f"{self.position} sees first {x.entity}, total {x.entities}")
                            if str(x.entity) == "wall":
                                self.color = color.blue
                                break
                            else:
                                self.color = color.red
                                break


    class Camera(Entity):   # The camera for viewing this project in action
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
                     self.up * 0.1)
            hit_info = raycast(origin, self.direction, ignore=(self, Entity, Unit, ground), distance=0.01, debug=False)
            if not hit_info.hit:
                if held_keys['c']:
                    self.position += self.direction * 3 * time.dt
                else:
                    self.position += self.direction * 10 * time.dt


    target_count = 9

    north = Wall(name='North', model='cube', collider='box', scale=(base_scale, base_scale, 0), color=color.white10)
    south = Wall(name='South', model='cube', collider='box', scale=(base_scale, base_scale, 0), color=color.white10)
    east = Wall(name='East', model='cube', collider='box', scale=(base_scale, base_scale, 0), rotation=(0, 90, 0), color=color.white10)
    west = Wall(name='West', model='cube', collider='box', scale=(base_scale, base_scale, 0), rotation=(0, 90, 0), color=color.white10)

    north.z = base_scale/2
    south.z = -base_scale/2
    east.x = base_scale/2
    west.x = -base_scale/2
    for i in range(target_count):

        swarm.append(Unit(model='cube', collider='box', origin_y=-.5, color=color.blue))
        swarm[i - 1].x = random.randint(round(-(base_scale-1)/2), round((base_scale-1)/2))
        swarm[i - 1].z = random.randint(round(-(base_scale-1)/2), round((base_scale-1)/2))
        swarm[i - 1].state.append(target_count)
        swarm[i - 1].state.append(time.time())

        targets.append(Target(model='cube', texture='vignette', scale=(0.75, 0.75, 0.75), collider='box', origin_y=-.5, color=color.violet))
        targets[i - 1].x = random.randint(round(-(base_scale-1)/2), round((base_scale-1)/2))
        targets[i - 1].z = random.randint(round(-(base_scale-1)/2), round((base_scale-1)/2))

    camera = Camera()
    # setting up a timer to record the speed of the units
    start_time = time.time()    # capture the time when the simulation begins
    timer = Text(text="Time: " + str(round(start_time, 3)), x=-0.98, y=0.48)   # print the timer in the top left screen
    Target_count = Text(text="Targets Remaining: " + str(target_count), x=-0.98, y=0.44)

    def update():
        global time_lapsed
        target_count = len(targets)
        swarm_count = len(swarm)
        if target_count > 0 and swarm_count > 0:
            # updating and printing the timer for use later in the machine learning
            end_time = time.time()  # capture the current time
            time_lapsed = end_time - start_time  # subtract the current time from the start time for the time elapsed
            timer.text = "Time: " + str(round(time_lapsed, 3)) + 's'  # print the timer

            Target_count.text = "Targets Remaining: " + str(target_count)
            for c in targets:
                for x in swarm:
                    if c.intersects(x):
                        x.success = True
                        c.touched = True
                        c.disable()
                        targets.remove(c)
                        swarm.remove(x)

                    if x.x > round((base_scale+1)/2) or x.x < -round((base_scale+1)/2) or x.z > round((base_scale+1)/2) or x.z < -round((base_scale+1)/2):
                        x.disable()
                        swarm.remove(x)

        if target_count == 0:
            Target_count.text = "All targets Found, press Shift + Q to exit"
            timer.text = "Total time elapsed: " + str(round(time_lapsed, 3)) + 's'
        elif swarm_count == 0:
            Target_count.text = "All swarm units disabled, press Shift + Q to exit"
            timer.text = "Total time elapsed: " + str(round(time_lapsed, 3)) + 's'
    # wall_left = Entity(model='cube', collider='box', scale_y=3, origin_y=-.5, color=color.azure, x=-4)
    # wall_right = duplicate(wall_left, x=4)
    app.run()
