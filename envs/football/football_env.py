import gfootball.env as gfootball_env
import gym
from collections import deque
import numpy as np
import enum
#%%  some commons


class Action(enum.IntEnum):
    Idle = 0
    Left = 1
    TopLeft = 2
    Top = 3
    TopRight = 4
    Right = 5
    BottomRight = 6
    Bottom = 7
    BottomLeft = 8
    LongPass = 9
    HighPass = 10
    ShortPass = 11
    Shot = 12
    Sprint = 13
    ReleaseDirection = 14
    ReleaseSprint = 15
    Slide = 16
    Dribble = 17
    ReleaseDribble = 18


sticky_index_to_action = [
    Action.Left,
    Action.TopLeft,
    Action.Top,
    Action.TopRight,
    Action.Right,
    Action.BottomRight,
    Action.Bottom,
    Action.BottomLeft,
    Action.Sprint,
    Action.Dribble
]

action_to_sticky_index = {
    a: index for index, a in enumerate(sticky_index_to_action)
}


class PlayerRole(enum.IntEnum):
    GoalKeeper = 0
    CenterBack = 1
    LeftBack = 2
    RightBack = 3
    DefenceMidfield = 4
    CentralMidfield = 5
    LeftMidfield = 6
    RIghtMidfield = 7
    AttackMidfield = 8
    CentralFront = 9


class GameMode(enum.IntEnum):
    Normal = 0
    KickOff = 1
    GoalKick = 2
    FreeKick = 3
    Corner = 4
    ThrowIn = 5
    Penalty = 6


#%%
"""
class FootballEnvDeveloping:
    def __init__(self):
        self.kick_actions = {
            Action.LongPass: 20,
            Action.HighPass: 28,
            Action.ShortPass: 36,
            Action.Shot: 44,
        }
        self.all_actions = [i for i in range(52) if i != 19]
        self.action_history = deque(maxlen=8)
        self.action_history.extend([0]*8)

        self.half_step = 1500

        self.env = env.create_environment(env_name="11_vs_11_kaggle",
                                          representation="raw",
                                          rewards="scoring,checkpoints",
                                          other_config_options={"action_set": "v2"})

        self.half_step = 1500
        self.reserved_action = None

    def reset(self):
        obs = self.env.reset()
        obs = obs[0]
        state = self._preprocess_obs(obs)
        return self._raw_obs_to_feature(state, self.action_history, self.half_step)

    def step(self, action: int):
        if self.reserved_action is not None:
            action = self.reserved_action
            self.reserved_action = None
        else:
            for a, index in self.kick_actions.items():
                if index <= action < index + 8:
                    action = a
                    self.reserved_action = Action(action - index + 1)

        self.action_history.append(action)
        obs, reward, done, info = self.env.step([action])
        obs = obs[0]
        state = self._preprocess_obs(obs)
        state = self._raw_obs_to_feature(obs, self.action_history, self.half_step)

        pass


    @staticmethod
    def _preprocess_obs(obs):
        mode = obs['game_mode']
        if mode == GameMode.FreeKick or \
                mode == GameMode.Corner or \
                mode == GameMode.Penalty or \
                mode == GameMode.GoalKick:
            # find nearest player and team
            def dist(xy1, xy2):
                return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

            team_player_position = [(0, i, p) for i, p in enumerate(obs['left_team'])] + \
                                   [(1, i, p) for i, p in enumerate(obs['right_team'])]
            distances = [(t[0], t[1], dist(t[2], obs['ball'][:2])) for t in team_player_position]
            distances = sorted(distances, key=lambda x: x[2])
            # print(mode, [t[2] for t in distances])
            # print(o['ball_owned_team'], o['ball_owned_player'], '->', distances[0][0], distances[0][1])
            # input()
            obs['ball_owned_team'] = distances[0][0]
            obs['ball_owned_player'] = distances[0][1]

        return obs


    def legal_actions(self, obs):
        all_actions = [i for i in range(52) if i != 19]
        # Illegal actions
        illegal_actions = set()
        # You have a ball?
        ball_owned_team = obs['ball_owned_team']
        if ball_owned_team != 0:  # not owned or free
            illegal_actions.add(int(Action.LongPass))
            illegal_actions.add(int(Action.HighPass))
            illegal_actions.add(int(Action.ShortPass))
            illegal_actions.add(int(Action.Shot))
            illegal_actions.add(int(Action.Dribble))
            for d in range(8):
                illegal_actions.add(self.kick_actions[Action.LongPass] + d)
                illegal_actions.add(self.kick_actions[Action.HighPass] + d)
                illegal_actions.add(self.kick_actions[Action.ShortPass] + d)
                illegal_actions.add(self.kick_actions[Action.Shot] + d)
        else:  # owned
            illegal_actions.add(int(Action.Slide))

        # Already sticky action?
        sticky_actions = obs['sticky_actions']
        if type(sticky_actions) == set:
            sticky_actions = [0] * 10

        if sticky_actions[action_to_sticky_index[Action.Sprint]] == 0:  # not action_sprint
            illegal_actions.add(int(Action.ReleaseSprint))

        if sticky_actions[action_to_sticky_index[Action.Dribble]] == 0:  # not action_dribble
            illegal_actions.add(int(Action.ReleaseDribble))

        if 1 not in sticky_actions[:8]:
            illegal_actions.add(int(Action.ReleaseDirection))

        return [a for a in all_actions if a not in illegal_actions]
"""


class TamakEriFeverEnvDeprecated(gym.Wrapper):
    """
    processing the raw obs according to the way of TamakEriFever
    current version: only support control one player, the designated player
    """
    def __init__(self, env):
        super().__init__(env)
        self.half_step = None
        self.num_left_players = None
        self.num_right_players = None

        self.all_actions = np.arange(19)
        self.action_history = deque(maxlen=8)
        self.action_history.extend([0] * 8)

    def reset(self):
        self.env.reset()
        # remove idle state
        obs, _, _, _ = self.env.step([0])
        obs = obs[0]

        self.half_step = obs["steps_left"] // 2
        self.num_left_players = obs["left_team"].shape[0]
        self.num_right_players = obs["right_team"].shape[0]

        state = self._preprocess_obs(obs)
        state = self._raw_obs_to_feature(state, self.action_history, self.half_step)
        legal_actions = self.legal_actions(obs)
        self.action_history.extend([0] * 8)
        return {"state": state,  'legal_actions': legal_actions}

    def step(self, action):
        self.action_history.append(action)
        obs, reward, done, info = self.env.step([action])
        obs = obs[0]
        info["score"] = obs["score"]
        state = self._preprocess_obs(obs)
        state = self._raw_obs_to_feature(state, self.action_history, self.half_step)
        legal_actions = self.legal_actions(obs)
        return {"state": state, "legal_actions": legal_actions}, reward, done, info

    @staticmethod
    def _preprocess_obs(obs):
        mode = obs['game_mode']
        if mode == GameMode.FreeKick or \
                mode == GameMode.Corner or \
                mode == GameMode.Penalty or \
                mode == GameMode.GoalKick:
            # find nearest player and team
            def dist(xy1, xy2):
                return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

            team_player_position = [(0, i, p) for i, p in enumerate(obs['left_team'])] + \
                                   [(1, i, p) for i, p in enumerate(obs['right_team'])]
            distances = [(t[0], t[1], dist(t[2], obs['ball'][:2])) for t in team_player_position]
            distances = sorted(distances, key=lambda x: x[2])
            # print(mode, [t[2] for t in distances])
            # print(o['ball_owned_team'], o['ball_owned_player'], '->', distances[0][0], distances[0][1])
            # input()
            obs['ball_owned_team'] = distances[0][0]
            obs['ball_owned_player'] = distances[0][1]

        return obs

    @staticmethod
    def _raw_obs_to_feature(obs: dict, action_history, half_step):
        """
        ・left players (x)
        ・left players (y)
        ・right players (x)
        ・right players (y)
        ・ball (x)
        ・ball (y)
        ・left goal (x)
        ・left goal (y)
        ・right goal (x)
        ・right goal (y)
        ・active (x)
        ・active (y)

        ・left players (x) - right players (x)
        ・left players (y) - right players (y)
        ・left players (x) - ball (x)
        ・left players (y) - ball (y)
        ・left players (x) - goal (x)
        ・left players (y) - goal (y)
        ・left players (x) - active (x)
        ・left players (y) - active (y)

        ・left players direction (x)
        ・left players direction (y)
        ・right players direction (x)
        ・right players direction (y)
        ・left players direction (x) - right players direction (x)
        ・left players direction (y) - right players direction (y)
        """

        # left players
        obs_left_team = obs['left_team']
        left_player_x = np.repeat(obs_left_team[:, 0][..., None], 11, axis=1)
        left_player_y = np.repeat(obs_left_team[:, 1][..., None], 11, axis=1)

        # right players
        obs_right_team = obs['right_team']
        right_player_x = np.repeat(obs_right_team[:, 0][..., None], 11, axis=1).transpose((1, 0))
        right_player_y = np.repeat(obs_right_team[:, 1][..., None], 11, axis=1).transpose((1, 0))

        # ball
        obs_ball = obs['ball']
        ball_x = np.ones((11, 11)) * obs_ball[0]
        ball_y = np.ones((11, 11)) * obs_ball[1]
        ball_z = np.ones((11, 11)) * obs_ball[2]

        # goal
        left_goal, right_goal = [-1, 0], [1, 0]
        left_goal_x = np.ones((11, 11)) * left_goal[0]
        left_goal_y = np.ones((11, 11)) * left_goal[1]
        right_goal_x = np.ones((11, 11)) * right_goal[0]
        right_goal_y = np.ones((11, 11)) * right_goal[1]

        # side line
        side_line_y = [-.42, .42]
        side_line_y_top = np.ones((11, 11)) * side_line_y[0]
        side_line_y_bottom = np.ones((11, 11)) * side_line_y[1]

        # active
        active = obs['active']
        active_player_x = np.repeat(obs_left_team[active][0][..., None, None], 11, axis=1).repeat(11, axis=0)
        active_player_y = np.repeat(obs_left_team[active][1][..., None, None], 11, axis=1).repeat(11, axis=0)

        # left players - right players
        left_minus_right_player_x = obs_left_team[:, 0][..., None] - obs_right_team[:, 0]
        left_minus_right_player_y = obs_left_team[:, 1][..., None] - obs_right_team[:, 1]

        # left players - ball
        left_minus_ball_x = (obs_left_team[:, 0][..., None] - obs_ball[0]).repeat(11, axis=1)
        left_minus_ball_y = (obs_left_team[:, 1][..., None] - obs_ball[1]).repeat(11, axis=1)

        # left players - right goal
        left_minus_right_goal_x = (obs_left_team[:, 0][..., None] - right_goal[0]).repeat(11, axis=1)
        left_minus_right_goal_y = (obs_left_team[:, 1][..., None] - right_goal[1]).repeat(11, axis=1)

        # left players - left goal
        left_minus_left_goal_x = (obs_left_team[:, 0][..., None] - left_goal[0]).repeat(11, axis=1)
        left_minus_left_goal_y = (obs_left_team[:, 1][..., None] - left_goal[1]).repeat(11, axis=1)

        # right players - right goal
        right_minus_right_goal_x = (obs_right_team[:, 0][..., None] - right_goal[0]).repeat(11, axis=1).transpose(
            (1, 0))
        right_minus_right_goal_y = (obs_right_team[:, 1][..., None] - right_goal[1]).repeat(11, axis=1).transpose(
            (1, 0))

        # right players - left goal
        right_minus_left_goal_x = (obs_right_team[:, 0][..., None] - left_goal[0]).repeat(11, axis=1).transpose(1, 0)
        right_minus_left_goal_y = (obs_right_team[:, 1][..., None] - left_goal[1]).repeat(11, axis=1).transpose(1, 0)

        # left players (x) - active
        left_minus_active_x = (obs_left_team[:, 0][..., None] - obs_left_team[active][0]).repeat(11, axis=1)
        left_minus_active_y = (obs_left_team[:, 1][..., None] - obs_left_team[active][1]).repeat(11, axis=1)

        # right player - ball
        right_minus_ball_x = (obs_right_team[:, 0][..., None] - obs_ball[0]).repeat(11, axis=1).transpose(1, 0)
        right_minus_ball_y = (obs_right_team[:, 1][..., None] - obs_ball[1]).repeat(11, axis=1).transpose(1, 0)

        # right player - active
        right_minus_active_x = (obs_right_team[:, 0][..., None] - obs_left_team[active][0]).repeat(11,
                                                                                                   axis=1).transpose(
            1, 0)
        right_minus_active_y = (obs_right_team[:, 1][..., None] - obs_left_team[active][1]).repeat(11,
                                                                                                   axis=1).transpose(
            1, 0)

        # left player - side line
        left_minus_side_top = np.abs(obs_left_team[:, 1][..., None] - side_line_y[0]).repeat(11, axis=1)
        left_minus_side_bottom = np.abs(obs_left_team[:, 1][..., None] - side_line_y[1]).repeat(11, axis=1)

        # right player - side line
        right_minus_side_top = np.abs(obs_right_team[:, 1][..., None] - side_line_y[0]).repeat(11,
                                                                                               axis=1).transpose(1,
                                                                                                                 0)
        right_minus_side_bottom = np.abs(obs_right_team[:, 1][..., None] - side_line_y[1]).repeat(11,
                                                                                                  axis=1).transpose(
            1, 0)

        # left players direction
        obs_left_team_direction = obs['left_team_direction']
        left_player_direction_x = np.repeat(obs_left_team_direction[:, 0][..., None], 11, axis=1)
        left_player_direction_y = np.repeat(obs_left_team_direction[:, 1][..., None], 11, axis=1)

        # right players direction
        obs_right_team_direction = obs['right_team_direction']
        right_player_direction_x = np.repeat(obs_right_team_direction[:, 0][..., None], 11, axis=1).transpose(1, 0)
        right_player_direction_y = np.repeat(obs_right_team_direction[:, 1][..., None], 11, axis=1).transpose(1, 0)

        # ball direction
        obs_ball_direction = obs['ball_direction']
        ball_direction_x = np.ones((11, 11)) * obs_ball_direction[0]
        ball_direction_y = np.ones((11, 11)) * obs_ball_direction[1]
        ball_direction_z = np.ones((11, 11)) * obs_ball_direction[2]

        # left players direction - right players direction
        left_minus_right_player_direction_x = obs_left_team_direction[:, 0][..., None] - obs_right_team_direction[:,
                                                                                         0]
        left_minus_right_player_direction_y = obs_left_team_direction[:, 1][..., None] - obs_right_team_direction[:,
                                                                                         1]

        # left players direction - ball direction
        left_minus_ball_direction_x = (obs_left_team_direction[:, 0][..., None] - obs_ball_direction[0]).repeat(11,
                                                                                                                axis=1)
        left_minus_ball_direction_y = (obs_left_team_direction[:, 1][..., None] - obs_ball_direction[1]).repeat(11,
                                                                                                                axis=1)

        # right players direction - ball direction
        right_minus_ball_direction_x = (obs_right_team_direction[:, 0][..., None] - obs_ball_direction[0]).repeat(
            11, axis=1).transpose(1, 0)
        right_minus_ball_direction_y = (obs_right_team_direction[:, 1][..., None] - obs_ball_direction[1]).repeat(
            11, axis=1).transpose(1, 0)

        # ball rotation
        obs_ball_rotation = obs['ball_rotation']
        ball_rotation_x = np.ones((11, 11)) * obs_ball_rotation[0]
        ball_rotation_y = np.ones((11, 11)) * obs_ball_rotation[1]
        ball_rotation_z = np.ones((11, 11)) * obs_ball_rotation[2]

        cnn_feature = np.stack([
            left_player_x,
            left_player_y,
            right_player_x,
            right_player_y,
            ball_x,
            ball_y,
            ball_z,
            left_goal_x,
            left_goal_y,
            right_goal_x,
            right_goal_y,
            side_line_y_top,
            side_line_y_bottom,
            active_player_x,
            active_player_y,
            left_minus_right_player_x,
            left_minus_right_player_y,
            left_minus_right_goal_x,
            left_minus_right_goal_y,
            left_minus_left_goal_x,
            left_minus_left_goal_y,
            right_minus_right_goal_x,
            right_minus_right_goal_y,
            right_minus_left_goal_x,
            right_minus_left_goal_y,
            left_minus_side_top,
            left_minus_side_bottom,
            right_minus_side_top,
            right_minus_side_bottom,
            right_minus_ball_x,
            right_minus_ball_y,
            right_minus_active_x,
            right_minus_active_y,
            left_minus_ball_x,
            left_minus_ball_y,
            left_minus_active_x,
            left_minus_active_y,
            ball_direction_x,
            ball_direction_y,
            ball_direction_z,
            left_minus_ball_direction_x,
            left_minus_ball_direction_y,
            right_minus_ball_direction_x,
            right_minus_ball_direction_y,
            left_player_direction_x,
            left_player_direction_y,
            right_player_direction_x,
            right_player_direction_y,
            left_minus_right_player_direction_x,
            left_minus_right_player_direction_y,
            ball_rotation_x,
            ball_rotation_y,
            ball_rotation_z,
        ], axis=0)

        # ball
        BALL_OWEND_1HOT = {-1: [0, 0], 0: [1, 0], 1: [0, 1]}
        ball_owned_team_ = obs['ball_owned_team']
        ball_owned_team = BALL_OWEND_1HOT[ball_owned_team_]  # {-1, 0, 1} None, self, opponent
        PLAYER_1HOT = np.concatenate([np.eye(11), np.zeros((1, 11))])
        ball_owned_player_ = PLAYER_1HOT[obs['ball_owned_player']]  # {-1, N-1}
        if ball_owned_team_ == -1:
            my_ball_owned_player = PLAYER_1HOT[-1]
            op_ball_owned_player = PLAYER_1HOT[-1]
        elif ball_owned_team_ == 0:
            my_ball_owned_player = ball_owned_player_
            op_ball_owned_player = PLAYER_1HOT[-1]
        else:
            my_ball_owned_player = PLAYER_1HOT[-1]
            op_ball_owned_player = ball_owned_player_

        ball_features = np.concatenate([
            obs['ball'],
            obs['ball_direction'],
            obs['ball_rotation']
        ]).astype(np.float32)

        # self team
        left_team_features = np.concatenate([
            [[1] for _ in obs['left_team']],  # left team flag
            obs['left_team'],  # position
            obs['left_team_direction'],
            [[v] for v in obs['left_team_tired_factor']],
            [[v] for v in obs['left_team_yellow_card']],
            [[v] for v in obs['left_team_active']],
            my_ball_owned_player[..., np.newaxis]
        ], axis=1).astype(np.float32)

        left_team_indice = np.arange(0, 11, dtype=np.int32)

        # opponent team
        right_team_features = np.concatenate([
            [[0] for _ in obs['right_team']],  # right team flag
            obs['right_team'],  # position
            obs['right_team_direction'],
            [[v] for v in obs['right_team_tired_factor']],
            [[v] for v in obs['right_team_yellow_card']],
            [[v] for v in obs['right_team_active']],
            op_ball_owned_player[..., np.newaxis]
        ], axis=1).astype(np.float32)

        right_team_indice = np.arange(0, 11, dtype=np.int32)

        # distance information
        def get_distance(xy1, xy2):
            return (((xy1 - xy2) ** 2).sum(axis=-1)) ** 0.5

        def get_line_distance(x1, x2):
            return np.abs(x1 - x2)

        def multi_scale(x, scale):
            return 2 / (1 + np.exp(-np.array(x)[..., np.newaxis] / np.array(scale)))

        both_team = np.concatenate([obs['left_team'], obs['right_team']], axis=0)
        ball = np.array([obs['ball'][:2]], dtype=np.float32)
        goal = np.array([[-1, 0], [1, 0]], dtype=np.float32)
        goal_line_x = np.array([-1, 1], dtype=np.float32)
        side_line_y = np.array([-.42, .42], dtype=np.float32)

        # ball <-> goal, goal line, side line distance
        b2g_distance = get_distance(ball, goal)
        b2gl_distance = get_line_distance(ball[0][0], goal_line_x)
        b2sl_distance = get_line_distance(ball[0][1], side_line_y)
        b2o_distance = np.concatenate([
            b2g_distance, b2gl_distance, b2sl_distance
        ], axis=-1)

        # player <-> ball, goal, back line, side line distance
        p2b_distance = get_distance(both_team[:, np.newaxis, :], ball[np.newaxis, :, :])
        p2g_distance = get_distance(both_team[:, np.newaxis, :], goal[np.newaxis, :, :])
        p2gl_distance = get_line_distance(both_team[:, :1], goal_line_x[np.newaxis, :])
        p2sl_distance = get_line_distance(both_team[:, 1:], side_line_y[np.newaxis, :])
        p2bo_distance = np.concatenate([
            p2b_distance, p2g_distance, p2gl_distance, p2sl_distance
        ], axis=-1)

        # player <-> player distance
        p2p_distance = get_distance(both_team[:, np.newaxis, :], both_team[np.newaxis, :, :])

        # apply Multiscale to distances
        # def concat_multiscale(x, scale):
        #    return np.concatenate([x[...,np.newaxis], 1 - multi_scale(x, scale)], axis=-1)

        # distance_scales = [.01, .05, .25, 1.25]
        # b2o_distance = 1 - multi_scale(b2o_distance, distance_scales).reshape(-1)
        # p2bo_distance = 1 - multi_scale(p2bo_distance, distance_scales).reshape(len(both_team), -1)
        # p2p_distance = 1 - multi_scale(p2p_distance, distance_scales).reshape(len(both_team), len(both_team), -1)

        # controlled player information
        control_flag_ = np.array(PLAYER_1HOT[obs['active']], dtype=np.float32)
        control_flag = np.concatenate([control_flag_, np.zeros(len(obs['right_team']))])[..., np.newaxis]

        # controlled status information
        DIR = [
            [-1, 0], [-.707, -.707], [0, 1], [.707, -.707],  # L, TL, T, TR
            [1, 0], [.707, .707], [0, -1], [-.707, .707]  # R, BR, B, BL
        ]
        sticky_direction = DIR[obs['sticky_actions'][:8].argmax()] if 1 in obs['sticky_actions'][:8] else [0, 0]
        sticky_flags = obs['sticky_actions'][8:]

        control_features = np.concatenate([
            sticky_direction,
            sticky_flags,
        ]).astype(np.float32)

        # Match state
        if obs['steps_left'] > half_step:
            steps_left_half = obs['steps_left'] - half_step
        else:
            steps_left_half = obs['steps_left']
        match_features = np.concatenate([
            multi_scale(obs['score'], [1, 3]).ravel(),
            multi_scale(obs['score'][0] - obs['score'][1], [1, 3]),
            multi_scale(obs['steps_left'], [10, 100, 1000, 10000]),
            multi_scale(steps_left_half, [10, 100, 1000, 10000]),
            ball_owned_team,
        ]).astype(np.float32)

        mode_index = np.array([obs['game_mode']], dtype=np.int32)

        # Super Mini Map
        # SMM_WIDTH = 96 #// 3
        # SMM_HEIGHT = 72 #// 3
        # SMM_LAYERS = ['left_team', 'right_team', 'ball', 'active']

        # # Normalized minimap coordinates
        # MINIMAP_NORM_X_MIN = -1.0
        # MINIMAP_NORM_X_MAX = 1.0
        # MINIMAP_NORM_Y_MIN = -1.0 / 2.25
        # MINIMAP_NORM_Y_MAX = 1.0 / 2.25

        # _MARKER_VALUE = 1  # 255

        # def get_smm_layers(config):
        #     return SMM_LAYERS

        # def mark_points(frame, points):
        #     """Draw dots corresponding to 'points'.
        #     Args:
        #       frame: 2-d matrix representing one SMM channel ([y, x])
        #       points: a list of (x, y) coordinates to be marked
        #     """
        #     for p in range(len(points) // 2):
        #         x = int((points[p * 2] - MINIMAP_NORM_X_MIN) /
        #                 (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame.shape[1])
        #         y = int((points[p * 2 + 1] - MINIMAP_NORM_Y_MIN) /
        #                 (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame.shape[0])
        #         x = max(0, min(frame.shape[1] - 1, x))
        #         y = max(0, min(frame.shape[0] - 1, y))
        #         frame[y, x] = _MARKER_VALUE

        # def generate_smm(observation, config=None,
        #                  channel_dimensions=(SMM_WIDTH, SMM_HEIGHT)):
        #     """Returns a list of minimap observations given the raw features for each
        #     active player.
        #     Args:
        #       observation: raw features from the environment
        #       config: environment config
        #       channel_dimensions: resolution of SMM to generate
        #     Returns:
        #       (N, H, W, C) - shaped np array representing SMM. N stands for the number of
        #       players we are controlling.
        #     """
        #     frame = np.zeros((len(observation), channel_dimensions[1],
        #                       channel_dimensions[0], len(get_smm_layers(config))),
        #                       dtype=np.uint8)

        #     for o_i, o in enumerate(observation):
        #         for index, layer in enumerate(get_smm_layers(config)):
        #             assert layer in o
        #             if layer == 'active':
        #                 if o[layer] == -1:
        #                     continue
        #                 mark_points(frame[o_i, :, :, index],
        #                             np.array(o['left_team'][o[layer]]).reshape(-1))
        #             else:
        #                 mark_points(frame[o_i, :, :, index], np.array(o[layer]).reshape(-1))
        #     return frame

        # smm = generate_smm([obs]).transpose(3, 1, 2, 0).squeeze(3).astype(np.float32)

        # ACTION_1HOT = np.eye(19)
        # action_history = np.stack([ACTION_1HOT[a] for a in action_history]).astype(np.float32)
        action_history = np.array(action_history, dtype=np.int32)[..., None]

        return {
            # features
            'ball': ball_features,
            'match': match_features,
            'player': {
                'self': left_team_features,
                'opp': right_team_features
            },
            'control': control_features,
            'player_index': {
                'self': left_team_indice,
                'opp': right_team_indice
            },
            'mode_index': mode_index,
            'control_flag': control_flag,
            # distances
            'distance': {
                'p2p': p2p_distance,
                'p2bo': p2bo_distance,
                'b2o': b2o_distance
            },
            # CNN
            'cnn_feature': cnn_feature,
            # SuperMiniMap
            # 'smm': smm,
            'action_history': action_history
        }

    def legal_actions(self, obs):
        # Illegal actions
        illegal_actions = set()
        # You have a ball?
        ball_owned_team = obs['ball_owned_team']
        if ball_owned_team != 0:  # not owned or free
            illegal_actions.add(int(Action.LongPass))
            illegal_actions.add(int(Action.HighPass))
            illegal_actions.add(int(Action.ShortPass))
            illegal_actions.add(int(Action.Shot))
            illegal_actions.add(int(Action.Dribble))
        else:  # owned
            illegal_actions.add(int(Action.Slide))

        # Already sticky action?
        sticky_actions = obs['sticky_actions']
        if type(sticky_actions) == set:
            sticky_actions = [0] * 10

        if sticky_actions[action_to_sticky_index[Action.Sprint]] == 0:  # not action_sprint
            illegal_actions.add(int(Action.ReleaseSprint))

        if sticky_actions[action_to_sticky_index[Action.Dribble]] == 0:  # not action_dribble
            illegal_actions.add(int(Action.ReleaseDribble))

        if 1 not in sticky_actions[:8]:
            illegal_actions.add(int(Action.ReleaseDirection))

        legal_actions = [a for a in self.all_actions if a not in illegal_actions]
        action_masks = np.zeros(19)
        action_masks[legal_actions] = 1
        return action_masks


class SimpleEnv(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.num_left_players = None
        self.num_right_players = None

        self.all_actions = np.arange(19)
        self.action_history = deque(maxlen=8)

    def reset(self):
        self.env.reset()
        # remove idle state
        obs, _, _, _ = self.env.step([0])
        obs = obs[0]

        self.action_history.extend([0] * 8)
        self.num_left_players = obs["left_team"].shape[0]
        self.num_right_players = obs["right_team"].shape[0]

        state = self._preprocess_obs(obs)
        state = self._raw_obs_to_feature(state, self.action_history)
        legal_actions = self.legal_actions(obs)

        return {"state": state, 'legal_actions': legal_actions}

    def step(self, action):
        self.action_history.append(action)
        obs, reward, done, info = self.env.step([action])
        obs = obs[0]
        info["score"] = obs["score"]
        state = self._preprocess_obs(obs)
        state = self._raw_obs_to_feature(state, self.action_history)
        legal_actions = self.legal_actions(obs)
        return {"state": state, "legal_actions": legal_actions}, reward, done, info

    @staticmethod
    def _preprocess_obs(obs):
        mode = obs['game_mode']
        if mode == GameMode.FreeKick or \
                mode == GameMode.Corner or \
                mode == GameMode.Penalty or \
                mode == GameMode.GoalKick:
            # find nearest player and team
            def dist(xy1, xy2):
                return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

            team_player_position = [(0, i, p) for i, p in enumerate(obs['left_team'])] + \
                                   [(1, i, p) for i, p in enumerate(obs['right_team'])]
            distances = [(t[0], t[1], dist(t[2], obs['ball'][:2])) for t in team_player_position]
            distances = sorted(distances, key=lambda x: x[2])
            # print(mode, [t[2] for t in distances])
            # print(o['ball_owned_team'], o['ball_owned_player'], '->', distances[0][0], distances[0][1])
            # input()
            obs['ball_owned_team'] = distances[0][0]
            obs['ball_owned_player'] = distances[0][1]

        return obs

    def legal_actions(self, obs):
        # Illegal actions
        illegal_actions = set()
        # You have a ball?
        ball_owned_team = obs['ball_owned_team']
        if ball_owned_team == 1:  # not owned
            illegal_actions.add(int(Action.LongPass))
            illegal_actions.add(int(Action.HighPass))
            illegal_actions.add(int(Action.ShortPass))
            illegal_actions.add(int(Action.Shot))
            illegal_actions.add(int(Action.Dribble))
        elif ball_owned_team == -1:  # free
            illegal_actions.add(int(Action.Dribble))
        elif ball_owned_team == 0:  # owned
            illegal_actions.add(int(Action.Slide))

        # Already sticky action?
        sticky_actions = obs['sticky_actions']
        if type(sticky_actions) == set:
            sticky_actions = [0] * 10

        if sticky_actions[action_to_sticky_index[Action.Sprint]] == 0:  # not action_sprint
            illegal_actions.add(int(Action.ReleaseSprint))

        if sticky_actions[action_to_sticky_index[Action.Dribble]] == 0:  # not action_dribble
            illegal_actions.add(int(Action.ReleaseDribble))

        if 1 not in sticky_actions[:8]:
            illegal_actions.add(int(Action.ReleaseDirection))

        legal_actions = [a for a in self.all_actions if a not in illegal_actions]
        action_masks = np.zeros(19)
        action_masks[legal_actions] = 1
        return action_masks

    def _raw_obs_to_feature(self, state, action_history=None):
        """
            follow these processing plus action_history
            'simple115'/'simple115v2': the observation is a vector of size 115.
                It holds:
                 - the ball_position and the ball_direction as (x,y,z)
                 - one hot encoding of who controls the ball.
                   [1, 0, 0]: nobody, [0, 1, 0]: left team, [0, 0, 1]: right team.
                 - one hot encoding of size 11 to indicate who is the active player
                   in the left team.
                 - 11 (x,y) positions for each player of the left team.
                 - 11 (x,y) motion vectors for each player of the left team.
                 - 11 (x,y) positions for each player of the right team.
                 - 11 (x,y) motion vectors for each player of the right team.
                 - one hot encoding of the game mode. Vector of size 7 with the
                   following meaning:
                   {NormalMode, KickOffMode, GoalKickMode, FreeKickMode,
                    CornerMode, ThrowInMode, PenaltyMode}.
                 Can only be used when the scenario is a flavor of normal game
            """
        feature = dict()
        feature["ball"] = np.concatenate([state["ball"], state["ball_direction"], state["ball_rotation"]],
                                                 axis=0)
        # global free: 0, left: 1-12, right: 12-23
        if state["ball_owned_team"] == -1:
             feature["ball_owned"] = 0
        else:
            feature["ball_owned"] = state["ball_owned_team"]*self.num_left_players + state["ball_owned_player"] + 1

        feature["player"] = np.concatenate([state["left_team"].ravel(), state["left_team_direction"].ravel(),
                                            state["right_team"].ravel(), state["right_team_direction"].ravel()], axis=0)

        feature["game_mode"] = state["game_mode"]

        feature["controlled_player_index"] = state["designated"]

        return feature


class TamakEriFeverEnv(gym.Wrapper):
    """
    processing the raw obs according to the way of TamakEriFever
    support control one player, the designated player
    """
    def __init__(self, env):
        super().__init__(env)
        self.num_left_players = None
        self.num_right_players = None
        self.ball_owned_one_hot = None
        self.player_left_one_hot = None
        self.player_right_one_hot = None
        self.direction = None
        self.all_actions = None
        self.action_history = deque(maxlen=8)
        self.half_step = None

    def reset(self):
        self.env.reset()
        # remove idle state
        obs, _, _, _ = self.env.step([0])
        obs = obs[0]

        self.num_left_players = obs["left_team"].shape[0]
        self.num_right_players = obs["right_team"].shape[0]
        self.ball_owned_one_hot = np.array([[1, 0], [0, 1], [0, 0]])
        self.player_left_one_hot = np.concatenate([np.eye(self.num_left_players),
                                                   np.zeros((1, self.num_left_players))],
                                                  axis=0)
        self.player_right_one_hot = np.concatenate([np.eye(self.num_right_players),
                                                    np.zeros((1, self.num_right_players))],
                                                   axis=0)
        self.direction = np.array([
            [-1, 0], [-.707, -.707], [0, 1], [.707, -.707],  # L, TL, T, TR
            [1, 0], [.707, .707], [0, -1], [-.707, .707]  # R, BR, B, BL
            ])
        self.all_actions = np.arange(self.env.action_space.n)
        self.action_history.extend([0]*8)
        self.half_step = obs["steps_left"] // 2

        obs = self._preprocess_obs(obs)
        feature = self._raw_obs_to_feature(obs)
        legal_actions = self.legal_actions(obs)

        return {"feature": feature,  'legal_actions': legal_actions}

    def step(self, action):
        self.action_history.append(action)
        obs, reward, done, info = self.env.step([action])
        obs = obs[0]
        info["score"] = obs["score"]

        obs = self._preprocess_obs(obs)
        feature = self._raw_obs_to_feature(obs)
        legal_actions = self.legal_actions(obs)

        return {"feature": feature, 'legal_actions': legal_actions}, reward, done, info

    @staticmethod
    def _preprocess_obs(obs):
        mode = obs['game_mode']
        if mode == GameMode.FreeKick or \
                mode == GameMode.Corner or \
                mode == GameMode.Penalty or \
                mode == GameMode.GoalKick:
            # find nearest player and team
            def dist(xy1, xy2):
                return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

            team_player_position = [(0, i, p) for i, p in enumerate(obs['left_team'])] + \
                                   [(1, i, p) for i, p in enumerate(obs['right_team'])]
            distances = [(t[0], t[1], dist(t[2], obs['ball'][:2])) for t in team_player_position]
            distances = sorted(distances, key=lambda x: x[2])
            # print(mode, [t[2] for t in distances])
            # print(o['ball_owned_team'], o['ball_owned_player'], '->', distances[0][0], distances[0][1])
            # input()
            obs['ball_owned_team'] = distances[0][0]
            obs['ball_owned_player'] = distances[0][1]

        return obs

    def _raw_obs_to_feature_to_do(self, obs: dict):
        ball_owned_team_ = obs["ball_owned_team"]
        ball_owned_team = self.ball_owned_one_hot[ball_owned_team_]  # {-1, 0, 1} None, self, opponent

        if ball_owned_team_ == -1:
            my_ball_owned_player = self.player_left_one_hot[-1]
            op_ball_owned_player = self.player_right_one_hot[-1]
        elif ball_owned_team_ == 0:
            my_ball_owned_player = self.player_left_one_hot[obs["ball_owned_player"]]
            op_ball_owned_player = self.player_right_one_hot[-1]
        else:
            my_ball_owned_player = self.player_left_one_hot[-1]
            op_ball_owned_player = self.player_right_one_hot[obs["ball_owned_player"]]


        # left players feature
        left_team_pos = obs['left_team']  # shape(11, 2)
        left_team_direction = obs['left_team_direction']  # shape(11, 2)
        left_team_features = np.concatenate([
            np.ones(shape=(self.num_left_players, 1)),  # left team flag
            left_team_pos,  # position
            left_team_direction,
            obs['left_team_tired_factor'][..., None],
            obs['left_team_yellow_card'][..., None],
            obs['left_team_active'][..., None],
            my_ball_owned_player[..., None]
        ], axis=-1).astype(np.float32)  # shape(11, 9)
        left_team_indice = np.arange(0, self.num_left_players, dtype=np.int32)

        # right players feature
        right_team_pos = obs['right_team']  # shape(11, 2)
        right_team_direction = obs['right_team_direction']  # shape(11, 2)
        right_team_features = np.concatenate([
            np.zeros(shape=(self.num_right_players, 1)),  # right team flag
            right_team_pos,  # position
            right_team_direction,
            obs['right_team_tired_factor'][..., None],
            obs['right_team_yellow_card'][..., None],
            obs['right_team_active'][..., None],
            op_ball_owned_player[..., None]
        ], axis=-1).astype(np.float32)  # shape (11,9)
        right_team_indice = np.arange(0, self.num_right_players, dtype=np.int32)


        # ball feature
        ball_pos = obs["ball"]  # shape(3)
        ball_direction = obs["ball_direction"]  # shape(3)
        ball_rotation = obs["ball_rotation"]   # shape(3)
        ball_features = np.concatenate([ball_pos, ball_direction, ball_rotation], axis=0).astype(np.float32)  # shape(9)

        # other feature
        goal = np.array([[-1, 0], [1, 0]], dtype=np.float32)
        side_line_y = np.array([-.42, .42], dtype=np.float32)

        # active player_feature
        active_player_pos = left_team_pos[obs['active']]  # shape(2)


        def create_relative_feature(team_feature, other_feature, reduction=True):
            tmp = team_feature[:, None, :] - other_feature[None, ...]
            if reduction:
                tmp = np.reshape(tmp, (team_feature.shape[0], -1))
            return tmp

        other_pos_features = np.concatenate([ball_pos[None, :2],
                                             goal,
                                             side_line_y,
                                             active_player_pos[None, ...]], axis=0)  # shape(5,2)

        left_minus_other_pos = create_relative_feature(left_team_pos, other_pos_features, reduction=True)   # shape(11,10)
        left_minus_other_direction = create_relative_feature(left_team_direction, ball_direction[None, :2])  # shape(11,2)
        right_minus_other_pos = create_relative_feature(right_team_pos, other_pos_features, reduction=True)  # shape(11,10)
        right_minus_other_direction = create_relative_feature(right_team_direction, ball_direction[None, :2])  # shape(11,2)

        left_minus_right_pos = create_relative_feature(left_team_pos, right_team_pos, reduction=False)  # shape(11,11,2)
        left_minus_right_direction = create_relative_feature(left_team_direction, right_team_direction,
                                                             reduction=False)  # shape(11,11,2)

        cnn_feature = np.concatenate([left_team_pos[:, None, :],
                                      left_team_direction[:, None, :],
                                      right_team_pos[None, ...],
                                      right_team_direction[None, ...],
                                      ball_features[None, None, ...],
                                      goal.ravel()[None, None, ...],
                                      side_line_y[None, None, ...],
                                      active_player_pos[None, None, ...],
                                      left_minus_other_pos[:, None, :],
                                      left_minus_other_direction[:, None, :],
                                      right_minus_other_pos[None, ...],
                                      right_minus_other_direction[None, ...],
                                      left_minus_right_pos,
                                      left_minus_right_direction], axis=-1)

        cnn_feature = cnn_feature.astype(np.float32)





        # controlled player information




        action_history = np.array(self.action_history, dtype=np.int32)[..., None]

        return {
            # features
            'ball': ball_features,
            'player': {
                'self': left_team_features,
                'opp': right_team_features
            },
            'player_index': {
                'self': left_team_indice,
                'opp': right_team_indice
            },

            # CNN
            'cnn_feature': cnn_feature,
            # SuperMiniMap
            # 'smm': smm,
            'action_history': action_history
        }

    def _get_control_feature(self, obs):
        control_flag_ = self.player_left_one_hot[obs['active']]
        control_flag = np.concatenate([control_flag_, np.zeros(self.num_right_players)]).astype(np.float32)

        # controlled status information

        sticky_direction = np.dot(obs['sticky_actions'][None, :8], self.direction).squeeze(axis=0)
        sticky_flags = obs['sticky_actions'][8:]

        control_features = np.concatenate([
            sticky_direction,
            sticky_flags,
        ]).astype(np.float32)

        return {'control_flag': control_flag, 'control': control_features}

    def _get_match_feature(self, obs):

        def multi_scale(x, scale):
            return 2 / (1 + np.exp(-np.array(x)[..., np.newaxis] / np.array(scale)))

        # Match state
        if obs['steps_left'] > self.half_step:
            steps_left_half = obs['steps_left'] - self.half_step
        else:
            steps_left_half = obs['steps_left']

        match_features = np.concatenate([
            multi_scale(obs['score'], [1, 3]).ravel(),  # 4
            multi_scale(obs['score'][0] - obs['score'][1], [1, 3]),  # 2
            multi_scale(obs['steps_left'], [10, 100, 1000, 10000]),  # 4
            multi_scale(steps_left_half, [10, 100, 1000, 10000]),  # 4
            self.ball_owned_one_hot[obs["ball_owned_team"]],  # 2
        ]).astype(np.float32)

        mode_index = np.array([obs['game_mode']], dtype=np.int32)

        return {"mode_index": mode_index, 'match': match_features}

    @staticmethod
    def _get_distance(obs):
        # distance information
        def get_distance(xy1, xy2):
            return (((xy1 - xy2) ** 2).sum(axis=-1)) ** 0.5

        def get_line_distance(x1, x2):
            return np.abs(x1 - x2)

        both_team = np.concatenate([obs['left_team'], obs['right_team']], axis=0)
        ball = obs['ball'][None, :2]
        ball = ball.astype(np.float32)
        goal = np.array([[-1, 0], [1, 0]], dtype=np.float32)
        goal_line_x = np.array([-1, 1], dtype=np.float32)
        side_line_y = np.array([-.42, .42], dtype=np.float32)

        # ball <-> goal, goal line, side line distance
        b2g_distance = get_distance(ball, goal)  # 2
        b2gl_distance = get_line_distance(ball[0][0], goal_line_x)  # 2
        b2sl_distance = get_line_distance(ball[0][1], side_line_y)  # 2
        b2o_distance = np.concatenate([
            b2g_distance, b2gl_distance, b2sl_distance
        ], axis=-1).astype(np.float32)  # 6

        # player <-> ball, goal, back line, side line distance
        p2b_distance = get_distance(both_team[:, np.newaxis, :], ball[np.newaxis, :, :])
        p2g_distance = get_distance(both_team[:, np.newaxis, :], goal[np.newaxis, :, :])
        p2gl_distance = get_line_distance(both_team[:, :1], goal_line_x[np.newaxis, :])
        p2sl_distance = get_line_distance(both_team[:, 1:], side_line_y[np.newaxis, :])
        p2bo_distance = np.concatenate([
            p2b_distance, p2g_distance, p2gl_distance, p2sl_distance
        ], axis=-1).astype(np.float32)

        # player <-> player distance
        p2p_distance = get_distance(both_team[:, np.newaxis, :], both_team[np.newaxis, :, :]).astype(np.float32)

        return {'distance': {
                            'p2p': p2p_distance,
                            'p2bo': p2bo_distance,
                            'b2o': b2o_distance
                            }
                }

    def _raw_obs_to_feature(self, obs: dict):
        # left players
        obs_left_team = obs['left_team']
        left_player_x = np.repeat(obs_left_team[:, 0][..., None], self.num_right_players, axis=1)
        left_player_y = np.repeat(obs_left_team[:, 1][..., None], self.num_right_players, axis=1)

        # right players
        obs_right_team = obs['right_team']
        right_player_x = np.repeat(obs_right_team[:, 0][..., None], self.num_left_players, axis=1).transpose((1, 0))
        right_player_y = np.repeat(obs_right_team[:, 1][..., None], self.num_left_players, axis=1).transpose((1, 0))

        # ball
        obs_ball = obs['ball']
        ball_x = np.ones((self.num_left_players, self.num_right_players)) * obs_ball[0]
        ball_y = np.ones((self.num_left_players, self.num_right_players)) * obs_ball[1]
        ball_z = np.ones((self.num_left_players, self.num_right_players)) * obs_ball[2]

        # goal
        left_goal, right_goal = [-1, 0], [1, 0]
        left_goal_x = np.ones((self.num_left_players, self.num_right_players)) * left_goal[0]
        left_goal_y = np.ones((self.num_left_players, self.num_right_players)) * left_goal[1]
        right_goal_x = np.ones((self.num_left_players, self.num_right_players)) * right_goal[0]
        right_goal_y = np.ones((self.num_left_players, self.num_right_players)) * right_goal[1]

        # side line
        side_line_y = [-.42, .42]
        side_line_y_top = np.ones((self.num_left_players, self.num_right_players)) * side_line_y[0]
        side_line_y_bottom = np.ones((self.num_left_players, self.num_right_players)) * side_line_y[1]

        # active
        active = obs['active']
        active_player_x = np.repeat(obs_left_team[active][0][..., None, None], self.num_right_players, axis=1).\
            repeat(self.num_left_players, axis=0)
        active_player_y = np.repeat(obs_left_team[active][1][..., None, None], self.num_right_players, axis=1).\
            repeat(self.num_left_players, axis=0)

        # left players - right players
        left_minus_right_player_x = obs_left_team[:, 0][..., None] - obs_right_team[:, 0]
        left_minus_right_player_y = obs_left_team[:, 1][..., None] - obs_right_team[:, 1]

        # left players - ball
        left_minus_ball_x = (obs_left_team[:, 0][..., None] - obs_ball[0]).repeat(self.num_right_players, axis=1)
        left_minus_ball_y = (obs_left_team[:, 1][..., None] - obs_ball[1]).repeat(self.num_right_players, axis=1)

        # left players - right goal
        left_minus_right_goal_x = (obs_left_team[:, 0][..., None] - right_goal[0]).repeat(self.num_right_players, axis=1)
        left_minus_right_goal_y = (obs_left_team[:, 1][..., None] - right_goal[1]).repeat(self.num_right_players, axis=1)

        # left players - left goal
        left_minus_left_goal_x = (obs_left_team[:, 0][..., None] - left_goal[0]).repeat(self.num_right_players, axis=1)
        left_minus_left_goal_y = (obs_left_team[:, 1][..., None] - left_goal[1]).repeat(self.num_right_players, axis=1)


        # right players - right goal
        right_minus_right_goal_x = (obs_right_team[:, 0][..., None] - right_goal[0]).\
            repeat(self.num_left_players, axis=1).transpose((1, 0))
        right_minus_right_goal_y = (obs_right_team[:, 1][..., None] - right_goal[1]).\
            repeat(self.num_left_players, axis=1).transpose((1, 0))

        # right players - left goal
        right_minus_left_goal_x = (obs_right_team[:, 0][..., None] - left_goal[0]).\
            repeat(self.num_left_players, axis=1).transpose(1, 0)
        right_minus_left_goal_y = (obs_right_team[:, 1][..., None] - left_goal[1]).\
            repeat(self.num_left_players, axis=1).transpose(1, 0)

        # left players (x) - active
        left_minus_active_x = (obs_left_team[:, 0][..., None] - obs_left_team[active][0]).\
            repeat(self.num_right_players, axis=1)
        left_minus_active_y = (obs_left_team[:, 1][..., None] - obs_left_team[active][1]).\
            repeat(self.num_right_players, axis=1)

        # right player - ball
        right_minus_ball_x = (obs_right_team[:, 0][..., None] - obs_ball[0]).\
            repeat(self.num_left_players, axis=1).transpose(1, 0)
        right_minus_ball_y = (obs_right_team[:, 1][..., None] - obs_ball[1]).\
            repeat(self.num_left_players, axis=1).transpose(1, 0)

        # right player - active
        right_minus_active_x = (obs_right_team[:, 0][..., None] - obs_left_team[active][0]).\
            repeat(self.num_left_players, axis=1).transpose(1, 0)
        right_minus_active_y = (obs_right_team[:, 1][..., None] - obs_left_team[active][1]).\
            repeat(self.num_left_players, axis=1).transpose(1, 0)

        # left player - side line
        left_minus_side_top = np.abs(obs_left_team[:, 1][..., None] - side_line_y[0]).\
            repeat(self.num_right_players, axis=1)
        left_minus_side_bottom = np.abs(obs_left_team[:, 1][..., None] - side_line_y[1]).\
            repeat(self.num_right_players, axis=1)

        # right player - side line
        right_minus_side_top = np.abs(obs_right_team[:, 1][..., None] - side_line_y[0]).\
            repeat(self.num_left_players, axis=1).transpose(1,0)
        right_minus_side_bottom = np.abs(obs_right_team[:, 1][..., None] - side_line_y[1]).\
            repeat(self.num_left_players, axis=1).transpose(1, 0)

        # left players direction
        obs_left_team_direction = obs['left_team_direction']
        left_player_direction_x = np.repeat(obs_left_team_direction[:, 0][..., None], self.num_right_players, axis=1)
        left_player_direction_y = np.repeat(obs_left_team_direction[:, 1][..., None], self.num_right_players, axis=1)

        # right players direction
        obs_right_team_direction = obs['right_team_direction']
        right_player_direction_x = np.repeat(obs_right_team_direction[:, 0][..., None], self.num_left_players, axis=1).\
            transpose(1, 0)
        right_player_direction_y = np.repeat(obs_right_team_direction[:, 1][..., None], self.num_left_players, axis=1).\
            transpose(1, 0)

        # ball direction
        obs_ball_direction = obs['ball_direction']
        ball_direction_x = np.ones((self.num_left_players, self.num_right_players)) * obs_ball_direction[0]
        ball_direction_y = np.ones((self.num_left_players, self.num_right_players)) * obs_ball_direction[1]
        ball_direction_z = np.ones((self.num_left_players, self.num_right_players)) * obs_ball_direction[2]

        # left players direction - right players direction
        left_minus_right_player_direction_x = obs_left_team_direction[:, 0][..., None] - obs_right_team_direction[:, 0]
        left_minus_right_player_direction_y = obs_left_team_direction[:, 1][..., None] - obs_right_team_direction[:, 1]

        # left players direction - ball direction
        left_minus_ball_direction_x = (obs_left_team_direction[:, 0][..., None] - obs_ball_direction[0])\
            .repeat(self.num_right_players, axis=1)
        left_minus_ball_direction_y = (obs_left_team_direction[:, 1][..., None] - obs_ball_direction[1])\
            .repeat(self.num_right_players, axis=1)

        # right players direction - ball direction
        right_minus_ball_direction_x = (obs_right_team_direction[:, 0][..., None] - obs_ball_direction[0]).repeat(
            self.num_left_players, axis=1).transpose(1, 0)
        right_minus_ball_direction_y = (obs_right_team_direction[:, 1][..., None] - obs_ball_direction[1]).repeat(
            self.num_left_players, axis=1).transpose(1, 0)

        # ball rotation
        obs_ball_rotation = obs['ball_rotation']
        ball_rotation_x = np.ones((self.num_left_players, self.num_right_players)) * obs_ball_rotation[0]
        ball_rotation_y = np.ones((self.num_left_players, self.num_right_players)) * obs_ball_rotation[1]
        ball_rotation_z = np.ones((self.num_left_players, self.num_right_players)) * obs_ball_rotation[2]

        cnn_feature = np.stack([
            left_player_x,
            left_player_y,
            right_player_x,
            right_player_y,
            ball_x,
            ball_y,
            ball_z,
            left_goal_x,
            left_goal_y,
            right_goal_x,
            right_goal_y,
            side_line_y_top,
            side_line_y_bottom,
            active_player_x,
            active_player_y,
            left_minus_right_player_x,
            left_minus_right_player_y,
            left_minus_right_goal_x,
            left_minus_right_goal_y,
            left_minus_left_goal_x,
            left_minus_left_goal_y,
            right_minus_right_goal_x,
            right_minus_right_goal_y,
            right_minus_left_goal_x,
            right_minus_left_goal_y,
            left_minus_side_top,
            left_minus_side_bottom,
            right_minus_side_top,
            right_minus_side_bottom,
            right_minus_ball_x,
            right_minus_ball_y,
            right_minus_active_x,
            right_minus_active_y,
            left_minus_ball_x,
            left_minus_ball_y,
            left_minus_active_x,
            left_minus_active_y,
            ball_direction_x,
            ball_direction_y,
            ball_direction_z,
            left_minus_ball_direction_x,
            left_minus_ball_direction_y,
            right_minus_ball_direction_x,
            right_minus_ball_direction_y,
            left_player_direction_x,
            left_player_direction_y,
            right_player_direction_x,
            right_player_direction_y,
            left_minus_right_player_direction_x,
            left_minus_right_player_direction_y,
            ball_rotation_x,
            ball_rotation_y,
            ball_rotation_z,
            ], axis=0)

        # ball
        BALL_OWEND_1HOT = {-1: [0, 0], 0: [1, 0], 1: [0, 1]}
        ball_owned_team_ = obs['ball_owned_team']
        ball_owned_team = BALL_OWEND_1HOT[ball_owned_team_]  # {-1, 0, 1} None, self, opponent
        PLAYER_1HOT_LEFT = np.concatenate([np.eye(self.num_left_players), np.zeros((1, self.num_left_players))], axis=0)
        PLAYER_1HOT_RIGHT = np.concatenate([np.eye(self.num_right_players), np.zeros((1, self.num_right_players))],
                                           axis=0)

        if ball_owned_team_ == -1:
            my_ball_owned_player = PLAYER_1HOT_LEFT[-1]
            op_ball_owned_player = PLAYER_1HOT_RIGHT[-1]
        elif ball_owned_team_ == 0:
            my_ball_owned_player = PLAYER_1HOT_LEFT[obs["ball_owned_player"]]
            op_ball_owned_player = PLAYER_1HOT_RIGHT[-1]
        else:
            my_ball_owned_player = PLAYER_1HOT_LEFT[-1]
            op_ball_owned_player = PLAYER_1HOT_RIGHT[obs["ball_owned_player"]]

        ball_features = np.concatenate([
            obs['ball'],
            obs['ball_direction'],
            obs['ball_rotation']
        ]).astype(np.float32)

        # self team
        left_team_features = np.concatenate([
            [[1] for _ in obs['left_team']],  # left team flag
            obs['left_team'],  # position
            obs['left_team_direction'],
            [[v] for v in obs['left_team_tired_factor']],
            [[v] for v in obs['left_team_yellow_card']],
            [[v] for v in obs['left_team_active']],
            my_ball_owned_player[..., np.newaxis]
        ], axis=1).astype(np.float32)

        left_team_indice = np.arange(0, self.num_left_players, dtype=np.int32)

        # opponent team
        right_team_features = np.concatenate([
            [[0] for _ in obs['right_team']],  # right team flag
            obs['right_team'],  # position
            obs['right_team_direction'],
            [[v] for v in obs['right_team_tired_factor']],
            [[v] for v in obs['right_team_yellow_card']],
            [[v] for v in obs['right_team_active']],
            op_ball_owned_player[..., np.newaxis]
        ], axis=1).astype(np.float32)

        right_team_indice = np.arange(0, self.num_right_players, dtype=np.int32)

        # distance information
        def get_distance(xy1, xy2):
            return (((xy1 - xy2) ** 2).sum(axis=-1)) ** 0.5

        def get_line_distance(x1, x2):
            return np.abs(x1 - x2)

        def multi_scale(x, scale):
            return 2 / (1 + np.exp(-np.array(x)[..., np.newaxis] / np.array(scale)))

        both_team = np.concatenate([obs['left_team'], obs['right_team']], axis=0)
        ball = np.array([obs['ball'][:2]], dtype=np.float32)
        goal = np.array([[-1, 0], [1, 0]], dtype=np.float32)
        goal_line_x = np.array([-1, 1], dtype=np.float32)
        side_line_y = np.array([-.42, .42], dtype=np.float32)

        # ball <-> goal, goal line, side line distance
        b2g_distance = get_distance(ball, goal)
        b2gl_distance = get_line_distance(ball[0][0], goal_line_x)
        b2sl_distance = get_line_distance(ball[0][1], side_line_y)
        b2o_distance = np.concatenate([
            b2g_distance, b2gl_distance, b2sl_distance
        ], axis=-1)

        # player <-> ball, goal, back line, side line distance
        p2b_distance = get_distance(both_team[:, np.newaxis, :], ball[np.newaxis, :, :])
        p2g_distance = get_distance(both_team[:, np.newaxis, :], goal[np.newaxis, :, :])
        p2gl_distance = get_line_distance(both_team[:, :1], goal_line_x[np.newaxis, :])
        p2sl_distance = get_line_distance(both_team[:, 1:], side_line_y[np.newaxis, :])
        p2bo_distance = np.concatenate([
            p2b_distance, p2g_distance, p2gl_distance, p2sl_distance
        ], axis=-1)

        # player <-> player distance
        p2p_distance = get_distance(both_team[:, np.newaxis, :], both_team[np.newaxis, :, :])

        # apply Multiscale to distances
        # def concat_multiscale(x, scale):
        #    return np.concatenate([x[...,np.newaxis], 1 - multi_scale(x, scale)], axis=-1)

        # distance_scales = [.01, .05, .25, 1.25]
        # b2o_distance = 1 - multi_scale(b2o_distance, distance_scales).reshape(-1)
        # p2bo_distance = 1 - multi_scale(p2bo_distance, distance_scales).reshape(len(both_team), -1)
        # p2p_distance = 1 - multi_scale(p2p_distance, distance_scales).reshape(len(both_team), len(both_team), -1)

        # controlled player information
        control_flag_ = np.array(PLAYER_1HOT_LEFT[obs['active']], dtype=np.float32)
        control_flag = np.concatenate([control_flag_, np.zeros(len(obs['right_team']))])[..., np.newaxis]

        # controlled status information
        DIR = [
            [-1, 0], [-.707, -.707], [0, 1], [.707, -.707],  # L, TL, T, TR
            [1, 0], [.707, .707], [0, -1], [-.707, .707]  # R, BR, B, BL
        ]
        sticky_direction = DIR[obs['sticky_actions'][:8].argmax()] if 1 in obs['sticky_actions'][:8] else [0, 0]
        sticky_flags = obs['sticky_actions'][8:]

        control_features = np.concatenate([
            sticky_direction,
            sticky_flags,
        ]).astype(np.float32)

        # Match state
        if obs['steps_left'] > self.half_step:
            steps_left_half = obs['steps_left'] - self.half_step
        else:
            steps_left_half = obs['steps_left']
        match_features = np.concatenate([
            multi_scale(obs['score'], [1, 3]).ravel(),
            multi_scale(obs['score'][0] - obs['score'][1], [1, 3]),
            multi_scale(obs['steps_left'], [10, 100, 1000, 10000]),
            multi_scale(steps_left_half, [10, 100, 1000, 10000]),
            ball_owned_team,
        ]).astype(np.float32)

        mode_index = np.array([obs['game_mode']], dtype=np.int32)

        # Super Mini Map
        # SMM_WIDTH = 96 #// 3
        # SMM_HEIGHT = 72 #// 3
        # SMM_LAYERS = ['left_team', 'right_team', 'ball', 'active']

        # # Normalized minimap coordinates
        # MINIMAP_NORM_X_MIN = -1.0
        # MINIMAP_NORM_X_MAX = 1.0
        # MINIMAP_NORM_Y_MIN = -1.0 / 2.25
        # MINIMAP_NORM_Y_MAX = 1.0 / 2.25

        # _MARKER_VALUE = 1  # 255

        # def get_smm_layers(config):
        #     return SMM_LAYERS

        # def mark_points(frame, points):
        #     """Draw dots corresponding to 'points'.
        #     Args:
        #       frame: 2-d matrix representing one SMM channel ([y, x])
        #       points: a list of (x, y) coordinates to be marked
        #     """
        #     for p in range(len(points) // 2):
        #         x = int((points[p * 2] - MINIMAP_NORM_X_MIN) /
        #                 (MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN) * frame.shape[1])
        #         y = int((points[p * 2 + 1] - MINIMAP_NORM_Y_MIN) /
        #                 (MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN) * frame.shape[0])
        #         x = max(0, min(frame.shape[1] - 1, x))
        #         y = max(0, min(frame.shape[0] - 1, y))
        #         frame[y, x] = _MARKER_VALUE

        # def generate_smm(observation, config=None,
        #                  channel_dimensions=(SMM_WIDTH, SMM_HEIGHT)):
        #     """Returns a list of minimap observations given the raw features for each
        #     active player.
        #     Args:
        #       observation: raw features from the environment
        #       config: environment config
        #       channel_dimensions: resolution of SMM to generate
        #     Returns:
        #       (N, H, W, C) - shaped np array representing SMM. N stands for the number of
        #       players we are controlling.
        #     """
        #     frame = np.zeros((len(observation), channel_dimensions[1],
        #                       channel_dimensions[0], len(get_smm_layers(config))),
        #                       dtype=np.uint8)

        #     for o_i, o in enumerate(observation):
        #         for index, layer in enumerate(get_smm_layers(config)):
        #             assert layer in o
        #             if layer == 'active':
        #                 if o[layer] == -1:
        #                     continue
        #                 mark_points(frame[o_i, :, :, index],
        #                             np.array(o['left_team'][o[layer]]).reshape(-1))
        #             else:
        #                 mark_points(frame[o_i, :, :, index], np.array(o[layer]).reshape(-1))
        #     return frame

        # smm = generate_smm([obs]).transpose(3, 1, 2, 0).squeeze(3).astype(np.float32)

        # ACTION_1HOT = np.eye(19)
        # action_history = np.stack([ACTION_1HOT[a] for a in action_history]).astype(np.float32)
        action_history = np.array(self.action_history, dtype=np.int32)[..., None]

        return {
            # features
            'ball': ball_features,
            'match': match_features,
            'player': {
                'self': left_team_features,
                'opp': right_team_features
            },
            'control': control_features,
            'player_index': {
                'self': left_team_indice,
                'opp': right_team_indice
            },
            'mode_index': mode_index,
            'control_flag': control_flag,
            # distances
            'distance': {
                'p2p': p2p_distance,
                'p2bo': p2bo_distance,
                'b2o': b2o_distance
            },
            # CNN
            'cnn_feature': cnn_feature,
            # SuperMiniMap
            # 'smm': smm,
            'action_history': action_history
        }

    def legal_actions(self, obs):
        # Illegal actions
        illegal_actions = set()
        # You have a ball?
        ball_owned_team = obs['ball_owned_team']
        if ball_owned_team == 1:  # opponent owned
            illegal_actions.add(int(Action.LongPass))
            illegal_actions.add(int(Action.HighPass))
            illegal_actions.add(int(Action.ShortPass))
            illegal_actions.add(int(Action.Shot))
            illegal_actions.add(int(Action.Dribble))
        elif ball_owned_team == -1:  # free
            illegal_actions.add(int(Action.Dribble))
        elif ball_owned_team == 0:  # owned
            illegal_actions.add(int(Action.Slide))

        # Already sticky action?
        sticky_actions = obs['sticky_actions']
        if type(sticky_actions) == set:
            sticky_actions = [0] * 10

        if sticky_actions[action_to_sticky_index[Action.Sprint]] == 0:  # not action_sprint
            illegal_actions.add(int(Action.ReleaseSprint))

        if sticky_actions[action_to_sticky_index[Action.Dribble]] == 0:  # not action_dribble
            illegal_actions.add(int(Action.ReleaseDribble))

        if 1 not in sticky_actions[:8]:
            illegal_actions.add(int(Action.ReleaseDirection))

        legal_actions = [a for a in self.all_actions if a not in illegal_actions]
        action_masks = np.zeros(19)
        action_masks[legal_actions] = 1
        return action_masks















