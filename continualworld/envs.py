from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import gym
import metaworld
import numpy as np
from gym.wrappers import TimeLimit

from continualworld.utils.wrappers import (CustomTimeLimit, OneHotAdder, 
                                           RandomizationWrapper, SuccessCounter)


def get_mt50() -> metaworld.MT50:
    """Returns a seeded instance of MT50 benchmark."""
    saved_random_state = np.random.get_state()
    np.random.seed(1)  # 使用固定种子以确保一致性
    try:
        mt50 = metaworld.MT50(seed=1)  # 添加种子参数
        np.random.set_state(saved_random_state)
        return mt50
    except Exception as e:
        print(f"Error initializing MT50: {e}")
        np.random.set_state(saved_random_state)
        raise


# 全局MT50实例
try:
    MT50 = get_mt50()
    META_WORLD_TIME_HORIZON = 200
    MT50_TASK_NAMES = list(MT50.train_classes.keys())  # 使用.keys()而不是直接使用字典
    MW_OBS_LEN = 12
    MW_ACT_LEN = 4
except Exception as e:
    print(f"Failed to initialize global MT50: {e}")
    raise


def get_task_name(name_or_number: Union[int, str]) -> str:
    try:
        index = int(name_or_number)
        return MT50_TASK_NAMES[index]
    except:
        return name_or_number


def set_simple_goal(env: gym.Env, name: str) -> None:
    goal = [task for task in MT50.train_tasks if task.env_name == name][0]
    env.set_task(goal)


def get_subtasks(name: str) -> List[metaworld.Task]:
    return [s for s in MT50.train_tasks if s.env_name == name]


def get_mt50_idx(env: gym.Env) -> int:
    idx = list(env._env_discrete_index.values())
    assert len(idx) == 1
    return idx[0]


def get_single_env(
    task: Union[int, str],
    one_hot_idx: int = 0,
    one_hot_len: int = 1,
    randomization: str = "random_init_all",
) -> gym.Env:
    """Returns a single task environment.

    Appends one-hot embedding to the observation, so that the model that operates on many envs
    can differentiate between them.

    Args:
      task: task name or MT50 number
      one_hot_idx: one-hot identifier (indicates order among different tasks that we consider)
      one_hot_len: length of the one-hot encoding, number of tasks that we consider
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.

    Returns:
      gym.Env: single-task environment
    """
    task_name = get_task_name(task)
    try:
        # 确保任务名称在训练类中
        if task_name not in MT50.train_classes:
            raise ValueError(f"Task {task_name} not found in MT50 train classes")
            
        # 创建环境实例
        env = MT50.train_classes[task_name]()
        
        # 设置任务
        task_instances = get_subtasks(task_name)
        if not task_instances:
            raise ValueError(f"No task instances found for {task_name}")
            
        env = RandomizationWrapper(env, task_instances, randomization)
        env = OneHotAdder(env, one_hot_idx=one_hot_idx, one_hot_len=one_hot_len)
        env = CustomTimeLimit(env, META_WORLD_TIME_HORIZON)
        env = SuccessCounter(env)
        env.name = task_name
        env.num_envs = 1
        return env
    except Exception as e:
        print(f"Error creating environment for task {task_name}: {e}")
        raise


def assert_equal_excluding_goal_dimensions(os1: gym.spaces.Box, os2: gym.spaces.Box) -> None:
    assert np.array_equal(os1.low[:9], os2.low[:9])
    assert np.array_equal(os1.high[:9], os2.high[:9])
    assert np.array_equal(os1.low[12:], os2.low[12:])
    assert np.array_equal(os1.high[12:], os2.high[12:])


def remove_goal_bounds(obs_space: gym.spaces.Box) -> None:
    obs_space.low[9:12] = -np.inf
    obs_space.high[9:12] = np.inf


class ContinualLearningEnv(gym.Env):
    def __init__(self, envs: List[gym.Env], steps_per_env: int) -> None:
        # 检查环境列表是否为空
        if not envs:
            raise ValueError("Environment list cannot be empty")
            
        # 保存第一个环境的动作空间和观察空间
        self.action_space = envs[0].action_space
        self.observation_space = deepcopy(envs[0].observation_space)
        remove_goal_bounds(self.observation_space)
        
        # 检查所有环境的观察空间结构是否兼容（忽略目标维度）
        for i in range(1, len(envs)):
            try:
                assert_equal_excluding_goal_dimensions(
                    envs[0].observation_space, envs[i].observation_space
                )
            except AssertionError:
                print(f"警告: 环境[0]和环境[{i}]的观察空间不完全兼容，但会尝试继续运行")
            
            # 如果动作空间不一致，打印警告信息
            if envs[0].action_space != envs[i].action_space:
                print(f"警告: 环境[0]和环境[{i}]的动作空间不同")
                print(f"环境[0]动作空间: {envs[0].action_space}")
                print(f"环境[{i}]动作空间: {envs[i].action_space}")
                print(f"将使用环境[0]的动作空间作为统一动作空间")

        self.envs = envs
        self.num_envs = len(envs)
        self.steps_per_env = steps_per_env
        self.steps_limit = self.num_envs * self.steps_per_env
        self.cur_step = 0
        self.cur_seq_idx = 0

    def _check_steps_bound(self) -> None:
        if self.cur_step >= self.steps_limit:
            raise RuntimeError("Steps limit exceeded for ContinualLearningEnv!")

    def pop_successes(self) -> List[bool]:
        all_successes = []
        self.avg_env_success = {}
        for env in self.envs:
            successes = env.pop_successes()
            all_successes += successes
            if len(successes) > 0:
                self.avg_env_success[env.name] = np.mean(successes)
        return all_successes

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        self._check_steps_bound()
        
        # 获取当前环境
        curr_env = self.envs[self.cur_seq_idx]
        
        # 检查当前环境的动作空间是否与基准环境不同，如果不同则适应动作
        if curr_env.action_space != self.action_space:
            # 如果是Box空间，尝试进行缩放
            if isinstance(curr_env.action_space, gym.spaces.Box) and isinstance(self.action_space, gym.spaces.Box):
                # 将动作从基准动作空间转换到当前环境的动作空间
                low_ratio = curr_env.action_space.low / self.action_space.low
                high_ratio = curr_env.action_space.high / self.action_space.high
                
                # 处理可能的除以零情况
                mask = (self.action_space.high - self.action_space.low) != 0
                if np.any(mask):
                    scale = np.ones_like(low_ratio)
                    scale[mask] = (curr_env.action_space.high[mask] - curr_env.action_space.low[mask]) / \
                                  (self.action_space.high[mask] - self.action_space.low[mask])
                    
                    # 归一化并重新缩放
                    norm_action = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
                    action = curr_env.action_space.low + norm_action * (curr_env.action_space.high - curr_env.action_space.low)
        
        # 执行动作
        obs, reward, done, info = curr_env.step(action)
        info["seq_idx"] = self.cur_seq_idx

        self.cur_step += 1
        if self.cur_step % self.steps_per_env == 0:
            # If we hit limit for current env, end the episode.
            # This may cause border episodes to be shorter than 200.
            done = True
            info["TimeLimit.truncated"] = True

            self.cur_seq_idx += 1

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        self._check_steps_bound()
        return self.envs[self.cur_seq_idx].reset()


def get_cl_env(
    tasks: List[Union[int, str]], steps_per_task: int, randomization: str = "random_init_all"
) -> gym.Env:
    """Returns continual learning environment.

    Args:
      tasks: list of task names or MT50 numbers
      steps_per_task: steps the agent will spend in each of single environments
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.

    Returns:
      gym.Env: continual learning environment
    """
    task_names = [get_task_name(task) for task in tasks]
    num_tasks = len(task_names)
    envs = []
    for i, task_name in enumerate(task_names):
        env = MT50.train_classes[task_name]()
        env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
        env = OneHotAdder(env, one_hot_idx=i, one_hot_len=num_tasks)
        env.name = task_name
        env = CustomTimeLimit(env, META_WORLD_TIME_HORIZON)
        env = SuccessCounter(env)
        envs.append(env)
    cl_env = ContinualLearningEnv(envs, steps_per_task)
    cl_env.name = "ContinualLearningEnv"
    return cl_env


class MultiTaskEnv(gym.Env):
    def __init__(
        self, envs: List[gym.Env], steps_per_env: int, cycle_mode: str = "episode"
    ) -> None:
        assert cycle_mode == "episode"
        
        # 检查环境列表是否为空
        if not envs:
            raise ValueError("Environment list cannot be empty")
            
        # 保存第一个环境的动作空间和观察空间
        self.action_space = envs[0].action_space
        self.observation_space = deepcopy(envs[0].observation_space)
        remove_goal_bounds(self.observation_space)
        
        # 检查所有环境的观察空间结构是否兼容（忽略目标维度）
        for i in range(1, len(envs)):
            try:
                assert_equal_excluding_goal_dimensions(
                    envs[0].observation_space, envs[i].observation_space
                )
            except AssertionError:
                print(f"警告: 环境[0]和环境[{i}]的观察空间不完全兼容，但会尝试继续运行")
            
            # 如果动作空间不一致，打印警告信息
            if envs[0].action_space != envs[i].action_space:
                print(f"警告: 环境[0]和环境[{i}]的动作空间不同")
                print(f"环境[0]动作空间: {envs[0].action_space}")
                print(f"环境[{i}]动作空间: {envs[i].action_space}")
                print(f"将使用环境[0]的动作空间作为统一动作空间")

        self.envs = envs
        self.num_envs = len(envs)
        self.steps_per_env = steps_per_env
        self.cycle_mode = cycle_mode

        self.steps_limit = self.num_envs * self.steps_per_env
        self.cur_step = 0
        self._cur_seq_idx = 0

    def _check_steps_bound(self) -> None:
        if self.cur_step >= self.steps_limit:
            raise RuntimeError("Steps limit exceeded for MultiTaskEnv!")

    def pop_successes(self) -> List[bool]:
        all_successes = []
        self.avg_env_success = {}
        for env in self.envs:
            successes = env.pop_successes()
            all_successes += successes
            if len(successes) > 0:
                self.avg_env_success[env.name] = np.mean(successes)
        return all_successes

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        self._check_steps_bound()
        
        # 获取当前环境
        curr_env = self.envs[self._cur_seq_idx]
        
        # 检查当前环境的动作空间是否与基准环境不同，如果不同则适应动作
        if curr_env.action_space != self.action_space:
            # 如果是Box空间，尝试进行缩放
            if isinstance(curr_env.action_space, gym.spaces.Box) and isinstance(self.action_space, gym.spaces.Box):
                # 将动作从基准动作空间转换到当前环境的动作空间
                low_ratio = curr_env.action_space.low / self.action_space.low
                high_ratio = curr_env.action_space.high / self.action_space.high
                
                # 处理可能的除以零情况
                mask = (self.action_space.high - self.action_space.low) != 0
                if np.any(mask):
                    scale = np.ones_like(low_ratio)
                    scale[mask] = (curr_env.action_space.high[mask] - curr_env.action_space.low[mask]) / \
                                  (self.action_space.high[mask] - self.action_space.low[mask])
                    
                    # 归一化并重新缩放
                    norm_action = (action - self.action_space.low) / (self.action_space.high - self.action_space.low)
                    action = curr_env.action_space.low + norm_action * (curr_env.action_space.high - curr_env.action_space.low)
        
        # 执行动作
        obs, reward, done, info = curr_env.step(action)
        info["mt_seq_idx"] = self._cur_seq_idx
        if self.cycle_mode == "step":
            self._cur_seq_idx = (self._cur_seq_idx + 1) % self.num_envs
        self.cur_step += 1

        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        self._check_steps_bound()
        if self.cycle_mode == "episode":
            self._cur_seq_idx = (self._cur_seq_idx + 1) % self.num_envs
        obs = self.envs[self._cur_seq_idx].reset()
        return obs


def get_mt_env(
    tasks: List[Union[int, str]], steps_per_task: int, randomization: str = "random_init_all"
):
    """Returns multi-task learning environment.

    Args:
      tasks: list of task names or MT50 numbers
      steps_per_task: agent will be limited to steps_per_task * len(tasks) steps
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.

    Returns:
      gym.Env: continual learning environment
    """
    task_names = [get_task_name(task) for task in tasks]
    num_tasks = len(task_names)
    envs = []
    for i, task_name in enumerate(task_names):
        env = MT50.train_classes[task_name]()
        env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
        env = OneHotAdder(env, one_hot_idx=i, one_hot_len=num_tasks)
        env.name = task_name
        env = CustomTimeLimit(env, META_WORLD_TIME_HORIZON)
        env = SuccessCounter(env)
        envs.append(env)
    mt_env = MultiTaskEnv(envs, steps_per_task)
    mt_env.name = "MultiTaskEnv"
    return mt_env
