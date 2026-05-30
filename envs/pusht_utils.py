import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
import gymnasium
from gymnasium import spaces
from typing import Dict, Sequence, Union, Optional, Tuple
import os
import gdown
import zarr

from pymunk.space_debug_draw_options import SpaceDebugColor

positive_y_is_up: bool = False

class DrawOptions(pymunk.SpaceDebugDrawOptions):
    def __init__(self, surface: pygame.Surface) -> None:
        self.surface = surface
        super(DrawOptions, self).__init__()

    def draw_circle(
        self,
        pos: Vec2d,
        angle: float,
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p = to_pygame(pos, self.surface)

        pygame.draw.circle(self.surface, fill_color.as_int(), p, round(radius), 0)
        pygame.draw.circle(self.surface, light_color(fill_color).as_int(), p, round(radius-4), 0)

    def draw_segment(self, a: Vec2d, b: Vec2d, color: SpaceDebugColor) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)
        pygame.draw.aalines(self.surface, color.as_int(), False, [p1, p2])

    def draw_fat_segment(
        self,
        a: Tuple[float, float],
        b: Tuple[float, float],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        p1 = to_pygame(a, self.surface)
        p2 = to_pygame(b, self.surface)

        r = round(max(1, radius * 2))
        pygame.draw.lines(self.surface, fill_color.as_int(), False, [p1, p2], r)
        if r > 2:
            orthog = [abs(p2[1] - p1[1]), abs(p2[0] - p1[0])]
            if orthog[0] == 0 and orthog[1] == 0:
                return
            scale = radius / (orthog[0] * orthog[0] + orthog[1] * orthog[1]) ** 0.5
            orthog[0] = round(orthog[0] * scale)
            orthog[1] = round(orthog[1] * scale)
            points = [
                (p1[0] - orthog[0], p1[1] - orthog[1]),
                (p1[0] + orthog[0], p1[1] + orthog[1]),
                (p2[0] + orthog[0], p2[1] + orthog[1]),
                (p2[0] - orthog[0], p2[1] - orthog[1]),
            ]
            pygame.draw.polygon(self.surface, fill_color.as_int(), points)
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p1[0]), round(p1[1])),
                round(radius),
            )
            pygame.draw.circle(
                self.surface,
                fill_color.as_int(),
                (round(p2[0]), round(p2[1])),
                round(radius),
            )

    def draw_polygon(
        self,
        verts: Sequence[Tuple[float, float]],
        radius: float,
        outline_color: SpaceDebugColor,
        fill_color: SpaceDebugColor,
    ) -> None:
        ps = [to_pygame(v, self.surface) for v in verts]
        ps += [ps[0]]
        pygame.draw.polygon(self.surface, light_color(fill_color).as_int(), ps)
        if radius > 0:
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                self.draw_fat_segment(a, b, radius, fill_color, fill_color)

    def draw_dot(
        self, size: float, pos: Tuple[float, float], color: SpaceDebugColor
    ) -> None:
        p = to_pygame(pos, self.surface)
        pygame.draw.circle(self.surface, color.as_int(), p, round(size), 0)

def to_pygame(p: Tuple[float, float], surface: pygame.Surface) -> Tuple[int, int]:
    if positive_y_is_up:
        return round(p[0]), surface.get_height() - round(p[1])
    else:
        return round(p[0]), round(p[1])

def light_color(color: SpaceDebugColor):
    color = np.minimum(1.2 * np.float32([color.r, color.g, color.b, color.a]), np.float32([255]))
    color = SpaceDebugColor(r=color[0], g=color[1], b=color[2], a=color[3])
    return color

def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

def farthest_point_sampling(points: np.ndarray, n_points: int, init_idx: int):
    assert(n_points >= 1)
    chosen_points = [points[init_idx]]
    for _ in range(n_points-1):
        cpoints = np.array(chosen_points)
        all_dists = np.linalg.norm(points[:,None,:] - cpoints[None,:,:], axis=-1)
        min_dists = all_dists.min(axis=1)
        next_idx = np.argmax(min_dists)
        next_pt = points[next_idx]
        chosen_points.append(next_pt)
    result = np.array(chosen_points)
    return result

from matplotlib import cm

class PymunkKeypointManager:
    def __init__(self, 
            local_keypoint_map: Dict[str, np.ndarray], 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        if color_map is None:
            cmap = cm.get_cmap('tab10')
            color_map = dict()
            for i, key in enumerate(local_keypoint_map.keys()):
                color_map[key] = (np.array(cmap.colors[i]) * 255).astype(np.uint8)

        self.local_keypoint_map = local_keypoint_map
        self.color_map = color_map

    @property
    def kwargs(self):
        return {
            'local_keypoint_map': self.local_keypoint_map,
            'color_map': self.color_map
        }

    @classmethod
    def create_from_pusht_env(cls, env, n_block_kps=9, n_agent_kps=3, seed=0, **kwargs):
        rng = np.random.default_rng(seed=seed)
        local_keypoint_map = dict()
        for name in ['block','agent']:
            # Create a temporary simulation to extract keypoints
            space = pymunk.Space()
            if name == 'agent':
                obj = env.add_circle(space, (256, 400), 15)
                n_kps = n_agent_kps
            else:
                obj = env.add_tee(space, (256, 300), 0)
                n_kps = n_block_kps

            screen = pygame.Surface((512,512))
            screen.fill(pygame.Color("white"))
            draw_options = DrawOptions(screen)
            space.debug_draw(draw_options)

            img = np.uint8(pygame.surfarray.array3d(screen).transpose(1, 0, 2))
            obj_mask = (img != np.array([255,255,255],dtype=np.uint8)).any(axis=-1)

            tf_img_obj = cls.get_tf_img_obj(obj)
            xy_img = np.moveaxis(np.array(np.indices((512,512))), 0, -1)[:,:,::-1]
            local_coord_img = tf_img_obj.inverse(xy_img.reshape(-1,2)).reshape(xy_img.shape)
            obj_local_coords = local_coord_img[obj_mask]

            # furthest point sampling
            init_idx = rng.choice(len(obj_local_coords))
            obj_local_kps = farthest_point_sampling(obj_local_coords, n_kps, init_idx)
            small_shift = rng.uniform(0, 1, size=obj_local_kps.shape)
            obj_local_kps += small_shift

            local_keypoint_map[name] = obj_local_kps

        return cls(local_keypoint_map=local_keypoint_map, **kwargs)

    @staticmethod
    def get_tf_img(pose: Sequence):
        pos = pose[:2]
        rot = pose[2]
        tf_img_obj = st.AffineTransform(
            translation=pos, rotation=rot)
        return tf_img_obj

    @classmethod
    def get_tf_img_obj(cls, obj: pymunk.Body):
        pose = tuple(obj.position) + (obj.angle,)
        return cls.get_tf_img(pose)

    def get_keypoints_global(self, 
            pose_map: Dict[str, Union[Sequence, pymunk.Body]], 
            is_obj=False):
        kp_map = dict()
        for key, value in pose_map.items():
            if is_obj:
                tf_img_obj = self.get_tf_img_obj(value)
            else:
                tf_img_obj = self.get_tf_img(value)
            kp_local = self.local_keypoint_map[key]
            kp_global = tf_img_obj(kp_local)
            kp_map[key] = kp_global
        return kp_map

    def draw_keypoints(self, img, kps_map, radius=1):
        scale = np.array(img.shape[:2]) / np.array([512,512])
        for key, value in kps_map.items():
            color = self.color_map[key].tolist()
            coords = (value * scale).astype(np.int32)
            for coord in coords:
                cv2.circle(img, coord, radius=radius, color=color, thickness=-1)
        return img


class PushTEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None,
            render_mode="rgb_array",
            rew_fn='sparse'
        ):
        self.render_mode = render_mode
        self._seed = None
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        self.k_p, self.k_v = 100, 20    # PD control gains.
        self.control_hz = self.metadata['render_fps']
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state
        self.prev_action = None
        
        self.rew_fn = rew_fn

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._seed = seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping

        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        if self.prev_action is None:
            self.prev_action = action
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt
                self.space.step(dt)

        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        done = coverage > self.success_threshold

        if self.rew_fn == 'sparse':
            reward = 0.0 if done else -1.0
        elif self.rew_fn == 'sparse_slow':
            reward = 0.0 if done else -1.0 * (1 + np.linalg.norm(action - self.prev_action)/200)
            self.prev_action = action
        elif self.rew_fn == 'l2':
            block_pos = np.array(self.block.position)
            goal_pos = self.goal_pose[:2]
            pos_dist = np.linalg.norm(block_pos - goal_pos)

            # Angle distance, handling wraparound.
            block_angle = self.block.angle % (2 * np.pi)
            goal_angle = self.goal_pose[2] % (2 * np.pi)
            angle_diff = abs(block_angle - goal_angle)
            angle_dist = min(angle_diff, 2 * np.pi - angle_diff)

            # Angle weighted by 50 to match the position scale.
            reward = -(pos_dist + 50 * angle_dist)
        elif self.rew_fn == 'l1':
            block_pos = np.array(self.block.position)
            goal_pos = self.goal_pose[:2]
            pos_dist = np.sum(np.abs(block_pos - goal_pos))

            # Angle distance, handling wraparound.
            block_angle = self.block.angle % (2 * np.pi)
            goal_angle = self.goal_pose[2] % (2 * np.pi)
            angle_diff = abs(block_angle - goal_angle)
            angle_dist = min(angle_diff, 2 * np.pi - angle_diff)

            # Angle weighted by 50 to match the position scale.
            reward = -(pos_dist + 50 * angle_dist)
        else:
            raise ValueError(f"Unknown reward function: {self.rew_fn}")

        terminated = bool(done)
        truncated = False  # Time limit is handled by the TimeLimit wrapper.

        observation = self._get_obs()
        info = self._get_info(coverage=coverage)
        info['success'] = 1.0 if done else 0.0
        info['reward'] = float(reward)

        return observation, float(reward), terminated, truncated, info

    def render(self):
        return self._render_frame(self.render_mode)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body

    def _get_info(self, coverage=None):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step,
            'coverage': coverage}
        return info

    def _render_frame(self, mode):
        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None

        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        if self.legacy:
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block
        self.space.step(1.0 / self.sim_hz)

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()

        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle(self.space, (256, 400), 15)
        self.block = self.add_tee(self.space, (256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handling
        self.space.on_collision(0, 0, post_solve=self._handle_collision)
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')
        return shape

    def add_circle(self, space, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        space.add(body, shape)
        return body

    def add_tee(self, space, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        space.add(body, shape1, shape2)
        return body

class PushTKeypointsEnv(PushTEnv):
    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            rew_fn='sparse',
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None,
            render_mode="rgb_array"
        ):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            rew_fn=rew_fn,
            render_action=render_action,
            render_mode=render_mode)
        ws = self.window_size

        if local_keypoint_map is None:
            # create default keypoint definition
            kp_kwargs = self.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs['local_keypoint_map']
            color_map = kp_kwargs['color_map']

        # create observation spaces
        Dblockkps = np.prod(local_keypoint_map['block'].shape)
        Dagentkps = np.prod(local_keypoint_map['agent'].shape)
        Dagentpos = 2

        Do = Dblockkps
        if agent_keypoints:
            # blockkp + agnet_pos
            Do += Dagentkps
        else:
            # blockkp + agnet_kp
            Do += Dagentpos
        # obs + obs_mask
        Dobs = Do * 2

        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        # mask range 0-1
        high[Do:] = 1.

        # (block_kps+agent_kps, xy+confidence)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float64
        )

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map,
            color_map=color_map)
        self.draw_kp_map = None

    @classmethod
    def genenerate_keypoint_manager_params(cls):
        env = PushTEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs

    def _get_obs(self):
        # get keypoints
        obj_map = {
            'block': self.block
        }
        if self.agent_keypoints:
            obj_map['agent'] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(
            pose_map=obj_map, is_obj=True)
        # python dict guerentee order of keys and values
        kps = np.concatenate(list(kp_map.values()), axis=0)

        # select keypoints to drop
        n_kps = kps.shape[0]
        rng = np.random.default_rng(self._seed)
        visible_kps = rng.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:,None], 2, axis=1)

        # save keypoints for rendering
        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {
            'block': vis_kps[:len(kp_map['block'])]
        }
        if self.agent_keypoints:
            draw_kp_map['agent'] = vis_kps[len(kp_map['block']):]
        self.draw_kp_map = draw_kp_map

        # construct obs
        obs = kps.flatten()
        obs_mask = kps_mask.flatten()
        if not self.agent_keypoints:
            # passing agent position when keypoints are not available
            agent_pos = np.array(self.agent.position)
            obs = np.concatenate([
                obs, agent_pos
            ])
            obs_mask = np.concatenate([
                obs_mask, np.ones((2,), dtype=bool)
            ])

        # obs, obs_mask
        obs = np.concatenate([
            obs, obs_mask.astype(obs.dtype)
        ], axis=0)
        return obs


    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints and img is not None:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0]/96))
        return img

def make_env(env_name, seed=None, rew_fn='sparse'):
    if env_name == 'pusht-keypoints-v0':
        env = PushTKeypointsEnv(render_mode="rgb_array", rew_fn=rew_fn)
    elif env_name == 'pusht-v0':
        env = PushTEnv(render_mode="rgb_array", rew_fn=rew_fn)
    else:
        raise ValueError(f"Unknown env name: {env_name}")

    if seed is not None:
        env.reset(seed=seed)

    # Add TimeLimit wrapper (300 steps is standard for PushT)
    from gymnasium.wrappers import TimeLimit
    env = TimeLimit(env, max_episode_steps=300)

    # Add Normalization wrapper
    env = PushTNormalizationWrapper(env)

    return env

class PushTNormalizationWrapper(gymnasium.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.min_val = 0.0
        self.max_val = 512.0

        # Update observation space
        low = np.full(self.observation_space.shape, -1.0, dtype=np.float32)
        high = np.full(self.observation_space.shape, 1.0, dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

        # Update action space
        low = np.full(self.action_space.shape, -1.0, dtype=np.float32)
        high = np.full(self.action_space.shape, 1.0, dtype=np.float32)
        self.action_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

    def normalize(self, x):
        return np.clip(2 * (x - self.min_val) / (self.max_val - self.min_val) - 1, -1.0, 1.0)

    def unnormalize(self, x):
        return (x + 1) / 2 * (self.max_val - self.min_val) + self.min_val

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.normalize(obs), info

    def step(self, action):
        action = self.unnormalize(action)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.normalize(obs), reward, terminated, truncated, info

def get_dataset(env_name):
    if env_name == 'pusht-keypoints-v0':
        dataset_path = os.path.join(os.path.expanduser('~/.ogpo/datasets/pusht'), 'pusht', 'pusht_cchi_v7_replay.zarr')
        if not os.path.exists(dataset_path):
            print(f"Downloading PushT dataset to {dataset_path}...")
            os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
            url = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
            zip_path = dataset_path + ".zip"
            import urllib.request
            urllib.request.urlretrieve(url, zip_path)
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(os.path.dirname(dataset_path)))
            os.remove(zip_path)

        print(f"Loading dataset from {dataset_path}")
        import zarr
        root = zarr.open(dataset_path, mode='r')

        data = {
            'observations': root['data']['state'][:],
            'actions': root['data']['action'][:],
            'terminals': root['data']['n_contacts'][:] > 0,
            'rewards': root['data']['n_contacts'][:]
        }

        # The dataset has no explicit RL terminals/rewards, so we regenerate
        # them: shift states for next_observations and use sparse rewards.
        states = root['data']['state'][:]
        actions = root['data']['action'][:]
        episode_ends = root['meta']['episode_ends'][:]

        observations = []
        next_observations = []
        act = []
        rew = []
        term = []
        masks = []

        # Convert raw states to keypoints via the env, then normalize.
        env = make_env(env_name)
        env.reset()
        kp_manager = env.unwrapped.kp_manager

        def normalize(x):
            return 2 * (x - 0.0) / (512.0 - 0.0) - 1

        print("Generating keypoints for dataset...")

        start_idx = 0
        for end_idx in episode_ends:
            episode_states = states[start_idx:end_idx]
            episode_actions = actions[start_idx:end_idx]

            for i in range(len(episode_states)):
                state = episode_states[i]
                action = episode_actions[i]

                # Set the env state to obtain the corresponding keypoint obs.
                env.unwrapped._set_state(state)
                obs = env.unwrapped._get_obs()

                obs = np.clip(normalize(obs), -1.0, 1.0)
                action = np.clip(normalize(action), -1.0, 1.0)

                observations.append(obs)
                act.append(action)

                if i < len(episode_states) - 1:
                    next_state = episode_states[i+1]
                    env.unwrapped._set_state(next_state)
                    next_obs = env.unwrapped._get_obs()
                    next_obs = np.clip(normalize(next_obs), -1.0, 1.0)

                    # Sparse reward: -1 per step, success handled at episode end.
                    reward = -1.0
                    done = False

                else:
                    # Last step: duplicate obs and assume the demonstration succeeded.
                    next_obs = obs.copy()
                    reward = 0.0
                    done = True

                next_observations.append(next_obs)
                rew.append(reward)
                term.append(done)
                masks.append(1.0 - float(done))

            start_idx = end_idx

        data['observations'] = np.array(observations)
        data['actions'] = np.array(act)
        data['next_observations'] = np.array(next_observations)
        data['rewards'] = np.array(rew)
        data['terminals'] = np.array(term)
        data['masks'] = np.array(masks)

        return data
    else:
        raise ValueError(f"Unknown env name: {env_name}")