from collections import deque
import torch
import random
import numpy as np
from typing import Union, Literal, List, Dict, Optional
import torch.nn.functional as F
import logging
from myrl.connection import MultiProcessJobExecutors
import queue


class MemoryReplay:
    def __init__(self, maxlen: int,  batch_size: int = 512, device=torch.device("cpu"), cached_in_device=True):
        """
        若device是cuda, cashed_in_device参数可以选择是否直接将样本存储到显存
        """
        self.memory = deque(maxlen=maxlen)
        self.device = device
        self.cached_in_device = cached_in_device
        self.batch_size = batch_size

    def cache(self, state, action, next_state, reward, done):
        """
        Store the experience to self.memory (replay buffer)
        the input must be np.array or int
        """
        state = torch.from_numpy(state).type(torch.float32)
        if type(action) in [int, np.int32, np.int64]:
            action = torch.tensor([action], dtype=torch.long)
        else:
            action = torch.from_numpy(action).type(torch.float32)

        next_state = torch.from_numpy(next_state).type(torch.float32)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([1. if done else 0.], dtype=torch.float32)

        if self.cached_in_device:
            self.memory.append(
                (state.to(self.device), action.to(self.device), next_state.to(self.device), reward.to(self.device),
                 done.to(self.device)))
        else:
            self.memory.append((state, action, next_state, reward, done))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, action, next_state, reward, done = map(torch.stack, zip(*batch))
        return self.to_device(state, action, next_state, reward, done)

    def all_sample(self):
        state, action, next_state, reward, done = map(torch.stack, zip(*self.memory))
        return self.to_device(state, action, next_state, reward, done)

    def empty(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

    def to_device(self, *args):
        if not self.cached_in_device:
            new_args = []
            for arg in args:
                new_args.append(arg.to(self.device))
            return new_args
        return args


class PriorityMemoryReplay(MemoryReplay):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.priority = deque(maxlen=self.memory.maxlen)
        self.alpha = 0.6  # the smoothing term of priority
        self.beta = 0.4  # the importance ratio of sample

    def cache(self, *args, **kwargs):
        super().cache(*args, **kwargs)
        # 新来的样本priority高
        max_priority = np.max(self.priority) if len(self.priority) > 0 else 1.
        self.priority.append(max_priority)

    def recall(self):
        temp = np.asarray(self.priority)
        temp = temp/np.sum(temp)

        index = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, replace=False, p=temp)
        # index type:np.ndarray

        batch = [self.memory[ind] for ind in index]
        selected_priority = torch.tensor([self.priority[ind] for ind in index], dtype=torch.float32, device=self.device)
        state, action, next_state, reward, done = map(torch.stack, zip(*batch))

        weight = torch.nn.functional.normalize(selected_priority.pow(-self.beta), p=1, dim=0)

        state, action, next_state, reward, done = self.to_device(state, action, next_state, reward, done)
        return state, action, next_state, reward, done, weight, index

    def update_priority(self, index, td_errors):
        # index is np.array
        # td_errors is np.array
        for ind, td_error in zip(index, td_errors):
            self.priority[ind] = td_error**self.alpha


class TrajReplay:
    def __init__(self,
                 maxlen: int,
                 device=torch.device("cpu"),
                 batch_size: int = 192,
                 forward_steps: Union[int, Literal["full"]] = 64,
                 max_forward_steps: Optional[int] = None,
                 burning_steps: int = 0
                 ):
        """
        :param maxlen: the length of episodes
        :param batch_size: the recall batch_size
        :param forward_steps: the length of each trajectory
        :param max_forward_steps: if forward_steps is full,
                                  this parameter determines the max length of episode,
                                  to avoid too long episode.
        :param burning_steps: burning steps for lstm.
        """
        self.episodes = deque(maxlen=maxlen)
        self.device = device
        self.batch_size = batch_size
        self.forward_steps = forward_steps
        self.max_forward_steps = max_forward_steps
        self.burning_steps = burning_steps

        self.num_cashed = 0

    def cache(self, episodes: Union[List[List[Dict]], List[Dict]]):
        if isinstance(episodes[0], list):
            self.episodes.extend(episodes)
            self.num_cashed += len(episodes)
        elif isinstance(episodes[0], dict):
            self.episodes.append(episodes)
            self.num_cashed += 1
        else:
            raise TypeError("episodes must be list of traj or traj, traj itself is a list of dict")

    def _process(self, episode, forward_steps):
        """

        :param episode: list of moment, a moment at least have observation, action, reward, done
        :param forward_steps: the sample steps
        :return:
        """
        train_st = np.random.randint(0, max(1, len(episode) - forward_steps + 1))
        st = max(0, train_st - self.burning_steps)
        ed = min(train_st + forward_steps, len(episode))
        episode = episode[st:ed]

        pad_num = forward_steps + self.burning_steps - (ed - st)

        observation = np.stack([moment["observation"] for moment in episode], axis=0)
        observation = np.pad(observation, ((0, pad_num), (0, 0)), mode="constant", constant_values=0)

        action = np.array([moment["action"] for moment in episode])
        action = np.pad(action, (0, pad_num), mode="constant", constant_values=0)

        reward = np.array([moment["reward"] for moment in episode])
        reward = np.pad(reward, (0, pad_num), mode="constant", constant_values=0)

        done = np.array([int(moment["done"]) for moment in episode])
        done = np.pad(done, (0, pad_num), mode="constant", constant_values=0)

        mask = np.zeros(forward_steps + self.burning_steps)
        mask[0: train_st - st] = 1
        mask[ed - st:] = 1

        tail_mask = np.zeros(forward_steps + self.burning_steps)
        tail_mask[0: train_st - st] = 1
        tail_mask[ed - st - 1:] = 1

        return observation, action, reward, done, mask, tail_mask

    def recall(self):

        observations, actions, rewards, dones, masks, tail_masks = \
            [], [], [], [], [], []

        indexes = np.random.choice(list(range(len(self.episodes))), size=self.batch_size, replace=False)
        if self.forward_steps == "full":
            max_traj_length = max([len(self.episodes[i]) for i in indexes])
            forward_steps = min(max_traj_length, self.max_forward_steps)
        else:
            forward_steps = self.forward_steps

        for i in indexes:
            observation, action, reward, done, mask, tail_mask = self._process(self.episodes[i], forward_steps)

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            masks.append(mask)
            tail_masks.append(tail_masks)

        return {"observations": torch.from_numpy(np.stack(observations)).to(self.device),
                #"behavior_log_probs": torch.from_numpy(np.stack(behavior_log_probs)).to(self.device),
                "actions": torch.from_numpy(np.stack(actions)).to(self.device),
                "rewards": torch.from_numpy(np.stack(rewards)).to(self.device),
                "masks": torch.from_numpy(np.stack(masks)).to(self.device),
                "dones": torch.from_numpy(np.stack(dones)).to(self.device),
                #"bootstrap_masks": torch.from_numpy(np.stack(bootstrap_masks)).to(self.device)
                }


class TrajList:
    def __init__(self,
                 device=torch.device("cpu")):

        self.episodes = []
        self.device = device

        self.num_cashed = 0

    def cache(self, episodes: Union[List[List[Dict]], List[Dict]]):
        if isinstance(episodes[0], list):
            self.episodes.extend(episodes)
            self.num_cashed += len(episodes)
        elif isinstance(episodes[0], dict):
            self.episodes.append(episodes)
            self.num_cashed += 1
        else:
            raise TypeError("episodes must be list of traj or traj, traj itself is a list of dict")

    def recall(self):
        observations, actions, rewards, dones, behavior_log_probs = [], [], [], [], []
        for episode in self.episodes:
            observation = np.stack([moment["observation"] for moment in episode], axis=0)
            action = np.array([moment["action"] for moment in episode])
            reward = np.array([moment["reward"] for moment in episode])
            done = np.array([moment["done"] for moment in episode])
            behavior_log_prob = np.array([moment["behavior_log_prob"] for moment in episode])

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            behavior_log_probs.append(behavior_log_prob)

        return {"observations": torch.from_numpy(np.stack(observations, axis=0)).type(torch.float32).to(self.device),
                "actions": torch.from_numpy(np.stack(actions, axis=0)).type(torch.int64).to(self.device),
                "rewards": torch.from_numpy(np.stack(rewards, axis=0)).type(torch.float32).to(self.device),
                "dones": torch.from_numpy(np.stack(dones, axis=0)).type(torch.float32).to(self.device),
                "behavior_log_probs": torch.from_numpy(np.stack(behavior_log_probs, axis=0)).type(torch.float32).to(self.device)
                }

    def empty(self):
        self.episodes = []

    def __len__(self):
        return len(self.episodes)


class TrajQueue:
    def __init__(self,
                 device=torch.device("cpu"),
                 batch_size=64):

        self.episodes = queue.Queue(maxsize=8)
        self.device = device
        self.batch_size = batch_size
        self.num_cashed = 0

    def cache(self, episode: List[Dict]):
        self.episodes.put(episode, timeout=0.1)
        self.num_cashed += 1
        logging.debug("put one episode")

    def recall(self):
        observations, actions, rewards, dones, behavior_log_probs = [], [], [], [], []
        while True:
            episode = self.episodes.get()
            observation = np.stack([moment["observation"] for moment in episode], axis=0)
            action = np.array([moment["action"] for moment in episode])
            reward = np.array([moment["reward"] for moment in episode])
            done = np.array([moment["done"] for moment in episode])
            behavior_log_prob = np.array([moment["behavior_log_prob"] for moment in episode])

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            behavior_log_probs.append(behavior_log_prob)

            if len(observations) == self.batch_size:
                break

        return {"observations": torch.from_numpy(np.stack(observations, axis=0)).type(torch.float32).to(self.device),
                "actions": torch.from_numpy(np.stack(actions, axis=0)).type(torch.int64).to(self.device),
                "rewards": torch.from_numpy(np.stack(rewards, axis=0)).type(torch.float32).to(self.device),
                "dones": torch.from_numpy(np.stack(dones, axis=0)).type(torch.float32).to(self.device),
                "behavior_log_probs": torch.from_numpy(np.stack(behavior_log_probs, axis=0)).type(torch.float32).to(self.device)
                }


class MultiProcessTrajQueue:
    def __init__(self,
                 maxlen: int,
                 device=torch.device("cpu"),
                 batch_size: int = 192,
                 num_batch_maker: int = 2
                 ):
        self.episodes = queue.Queue(maxsize=maxlen)
        self.device = device
        self.batch_size = batch_size

        self.num_cached = 0

        self.batch_maker = MultiProcessJobExecutors(func=make_batch, send_generator=self.send_raw_batch(),
                                                    postprocess=self.post_process,
                                                    num=num_batch_maker, buffer_length=1, num_receivers=1,
                                                    name_prefix="batch_maker",
                                                    logger_file_path="./log/impala_batch_maker.txt")

    def cache(self, episode):
        try:
            self.episodes.put(episode, timeout=0.1)
            self.num_cached += 1
            logging.debug("put one episode")
        except queue.Full:
            logging.critical(" generate rate is larger than consumer")
            raise

    def recall(self):
        return self.batch_maker.recv()

    def send_raw_batch(self):
        while True:
            yield [self.episodes.get() for _ in range(self.batch_size)]

    def start(self):
        self.batch_maker.start()

    def post_process(self, batch):
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)
        return batch


class MultiProcessBatcher:
    def __init__(self,
                 maxlen: int,
                 device=torch.device("cpu"),
                 batch_size: int = 192,
                 forward_steps: int = 64,
                 num_batch_maker: int = 2
                 ):
        self.episodes = deque(maxlen=maxlen)
        self.device = device
        self.batch_size = batch_size
        self.forward_steps = forward_steps

        self.num_cached = 0

        self.batch_maker = MultiProcessJobExecutors(func=make_batch, send_generator=self.send_raw_batch(),
                                                    postprocess=self.post_process,
                                                    num=num_batch_maker, buffer_length=1, num_receivers=1,
                                                    name_prefix="batch_maker",
                                                    logger_file_path="./log/impala_batch_maker.txt")

    def cache(self, episodes: Union[List[List[Dict]], List[Dict]]):
        if isinstance(episodes[0], list):
            self.episodes.extend(episodes)
            self.num_cached += len(episodes)
            logging.debug(f"total cached episodes is {self.num_cached}")
        elif isinstance(episodes[0], dict):
            self.episodes.append(episodes)
            self.num_cached += 1
            logging.debug(f"total cached episodes is {self.num_cached}")
        else:
            raise TypeError("episodes must be list of traj or traj, traj itself is a list of dict")

    def recall(self):
        return self.batch_maker.recv()

    def send_raw_batch(self):
        while True:
            yield [self.send_a_sample() for _ in range(self.batch_size)]

    def send_a_sample(self):
        while True:
            ep_idx = random.randrange(len(self.episodes))
            accept_rate = 1 - (len(self.episodes) - 1 - ep_idx) / self.episodes.maxlen
            if random.random() < accept_rate:
                ep = self.episodes[ep_idx]
                st = random.randrange(
                    1 + max(0, len(ep) - self.forward_steps))  # change start turn by sequence length
                return ep[st:st+self.forward_steps]

    def start(self):
        self.batch_maker.start()

    def post_process(self, batch):
        for key in batch.keys():
            batch[key] = batch[key].to(self.device)
        return batch

    def __len__(self):
        return len(self.episodes)


def make_batch(episodes):
    observations, actions, rewards, dones, behavior_log_probs = [], [], [], [], []
    for episode in episodes:
        observation = np.stack([moment["observation"] for moment in episode], axis=0)
        action = np.array([moment["action"] for moment in episode])
        reward = np.array([moment["reward"] for moment in episode])
        done = np.array([moment["done"] for moment in episode])
        behavior_log_prob = np.array([moment["behavior_log_prob"] for moment in episode])

        observations.append(observation)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        behavior_log_probs.append(behavior_log_prob)

    return {"observations": torch.from_numpy(np.stack(observations, axis=0)).type(torch.float32),
            "actions": torch.from_numpy(np.stack(actions, axis=0)).type(torch.int64),
            "rewards": torch.from_numpy(np.stack(rewards, axis=0)).type(torch.float32),
            "dones": torch.from_numpy(np.stack(dones, axis=0)).type(torch.float32),
            "behavior_log_probs": torch.from_numpy(np.stack(behavior_log_probs, axis=0)).type(torch.float32)
            }
