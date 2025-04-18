import random
from typing import Dict
from collections import deque
import numpy as np
import tensorflow as tf

def proportional(nsample, buf_sizes):
    T = np.sum(buf_sizes)
    S = nsample
    sample_sizes = np.zeros(len(buf_sizes), dtype=np.int64)
    for i in range(len(buf_sizes)):
        if S < 1:
            break
        sample_sizes[i] = int(round(S * buf_sizes[i] / T))
        T -= buf_sizes[i]
        S -= sample_sizes[i]
    assert sum(sample_sizes) == nsample, str(sum(sample_sizes))+" and "+str(nsample)
    return sample_sizes
    
class ReplayBufferFIFO(object):
    def __init__(self, size):
        """
        Implements a ring buffer (FIFO).
        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.
        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer
        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
            old_data = None
        else:
            old_data = self._storage[self._next_idx]
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        
        return old_data # used in MultiTimescale buffer
        
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.
        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)



class ReplayBuffer:
    """A simple FIFO experience replay buffer for SAC agents."""

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.actions_buf[self.ptr] = action
        self.rewards_buf[self.ptr] = reward
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
        )


class ReservoirReplayBuffer(ReplayBuffer):
    """Buffer for SAC agents implementing reservoir sampling."""

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        super().__init__(obs_dim, act_dim, size)
        self.timestep = 0

    def store(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        current_t = self.timestep
        self.timestep += 1

        if current_t < self.max_size:
            buffer_idx = current_t
        else:
            buffer_idx = random.randint(0, current_t)
            if buffer_idx >= self.max_size:
                return

        self.obs_buf[buffer_idx] = obs
        self.next_obs_buf[buffer_idx] = next_obs
        self.actions_buf[buffer_idx] = action
        self.rewards_buf[buffer_idx] = reward
        self.done_buf[buffer_idx] = done
        self.size = min(self.size + 1, self.max_size)


class ExpertReplayBuffer:
    """A expert experience replay buffer for behavioral cloning,
    which does not support overwriting old samples."""

    def __init__(self, obs_dim: int, act_dim: int, size: int) -> None:
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.policy_mu_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.policy_std_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.q1_buf = np.zeros([size,], dtype=np.float32)
        self.q2_buf = np.zeros([size,], dtype=np.float32)
        self.size, self.max_size = 0, size

    def store(self, obs: np.ndarray, actions: np.ndarray, policy_mu: np.ndarray, policy_std: np.ndarray,
              q1: np.ndarray, q2: np.ndarray) -> None:
        assert self.size + obs.shape[0] <= self.max_size
        range_start = self.size
        range_end = self.size + obs.shape[0]
        self.obs_buf[range_start:range_end] = obs
        self.actions_buf[range_start:range_end] = actions
        self.policy_mu_buf[range_start:range_end] = policy_mu
        self.policy_std_buf[range_start:range_end] = policy_std
        self.q1_buf[range_start:range_end] = q1
        self.q2_buf[range_start:range_end] = q2
        self.size = range_end

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            policy_mu=tf.convert_to_tensor(self.policy_mu_buf[idxs]),
            policy_std=tf.convert_to_tensor(self.policy_std_buf[idxs]),
            q1=tf.convert_to_tensor(self.q1_buf[idxs]),
            q2=tf.convert_to_tensor(self.q2_buf[idxs]),
        )


class PerfectReplayBuffer:
    """A simple Perfect replay buffer for SAC agents."""

    def __init__(self, obs_dim: int, act_dim: int, size: int, steps_per_task: int) -> None:
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rewards_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.steps_per_task, self.size, self.max_size = steps_per_task, 0, size

    def store(
        self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
    ) -> None:
        assert self.size < self.max_size, "Out of perfect memory!"
        self.obs_buf[self.size] = obs
        self.next_obs_buf[self.size] = next_obs
        self.actions_buf[self.size] = action
        self.rewards_buf[self.size] = reward
        self.done_buf[self.size] = done
        self.size = self.size + 1

    def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
        cur_task_t = self.size % self.steps_per_task
        cur_task_idx = self.size // self.steps_per_task

        idxs = np.random.randint(self.size - cur_task_t, self.size, size=batch_size)
        if cur_task_idx > 0:
            idxs = np.append(idxs, np.random.randint(0, self.size - cur_task_t, size=batch_size))

        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
        )

    # def sample_cur_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
    #     cur_task_t = self.size % self.steps_per_task
    #     idxs = np.random.randint(self.size - cur_task_t, self.size, size=batch_size)
    #
    #     return dict(
    #         obs=tf.convert_to_tensor(self.obs_buf[idxs]),
    #         next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
    #         actions=tf.convert_to_tensor(self.actions_buf[idxs]),
    #         rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
    #         done=tf.convert_to_tensor(self.done_buf[idxs]),
    #     )
    #
    # def sample_his_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
    #     cur_task_t = self.size % self.steps_per_task
    #     idxs = np.random.randint(0, self.size - cur_task_t, size=batch_size)
    #
    #     return dict(
    #         obs=tf.convert_to_tensor(self.obs_buf[idxs]),
    #         next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
    #         actions=tf.convert_to_tensor(self.actions_buf[idxs]),
    #         rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
    #         done=tf.convert_to_tensor(self.done_buf[idxs]),
    #     )

    def sample_cur_batch_for_bc(self, batch_size: int) -> Dict[str, tf.Tensor]:
        idxs = np.random.randint(self.size - self.steps_per_task, self.size, size=batch_size)

        return dict(
            obs=tf.convert_to_tensor(self.obs_buf[idxs]),
            next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
            actions=tf.convert_to_tensor(self.actions_buf[idxs]),
            rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
            done=tf.convert_to_tensor(self.done_buf[idxs]),
        )


# class PerfectReplayBuffer_:
#     """A simple Perfect replay buffer for SAC agents."""
#
#     def __init__(self, obs_dim: int, act_dim: int, size: int, steps_per_task: int) -> None:
#         self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         self.actions_buf = np.zeros([size, act_dim], dtype=np.float32)
#         self.rewards_buf = np.zeros(size, dtype=np.float32)
#         self.done_buf = np.zeros(size, dtype=np.float32)
#
#         self.policy_mu_buf = np.zeros([size, act_dim], dtype=np.float32)
#         self.policy_std_buf = np.zeros([size, act_dim], dtype=np.float32)
#         self.q1_buf = np.zeros([size,], dtype=np.float32)
#         self.q2_buf = np.zeros([size,], dtype=np.float32)
#
#         self.steps_per_task, self.size, self.max_size = steps_per_task, 0, size
#
#     def store(
#         self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool
#     ) -> None:
#         assert self.size < self.max_size, "Out of perfect memory!"
#         self.obs_buf[self.size] = obs
#         self.next_obs_buf[self.size] = next_obs
#         self.actions_buf[self.size] = action
#         self.rewards_buf[self.size] = reward
#         self.done_buf[self.size] = done
#         self.size = self.size + 1
#
#     def store_target(self, policy_mu: np.ndarray, policy_std: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> None:
#         self.policy_mu_buf = policy_mu
#         self.policy_std_buf = policy_std
#         self.q1_buf = q1
#         self.q2_buf = q2
#
#     def sample_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
#         cur_task_t = self.size % self.steps_per_task
#         cur_task_idx = self.size // self.steps_per_task
#
#         idxs = np.random.randint(self.size - cur_task_t, self.size, size=batch_size)
#         if cur_task_idx > 0:
#             idxs = np.append(idxs, np.random.randint(0, self.size - cur_task_t, size=batch_size))
#
#         return dict(
#             obs=tf.convert_to_tensor(self.obs_buf[idxs]),
#             next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
#             actions=tf.convert_to_tensor(self.actions_buf[idxs]),
#             rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
#             done=tf.convert_to_tensor(self.done_buf[idxs]),
#         )
#
#     # def sample_cur_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
#     #     cur_task_t = self.size % self.steps_per_task
#     #     idxs = np.random.randint(self.size - cur_task_t, self.size, size=batch_size)
#     #
#     #     return dict(
#     #         obs=tf.convert_to_tensor(self.obs_buf[idxs]),
#     #         next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
#     #         actions=tf.convert_to_tensor(self.actions_buf[idxs]),
#     #         rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
#     #         done=tf.convert_to_tensor(self.done_buf[idxs]),
#     #     )
#     #
#     # def sample_his_batch(self, batch_size: int) -> Dict[str, tf.Tensor]:
#     #     cur_task_t = self.size % self.steps_per_task
#     #     idxs = np.random.randint(0, self.size - cur_task_t, size=batch_size)
#     #
#     #     return dict(
#     #         obs=tf.convert_to_tensor(self.obs_buf[idxs]),
#     #         next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
#     #         actions=tf.convert_to_tensor(self.actions_buf[idxs]),
#     #         rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
#     #         done=tf.convert_to_tensor(self.done_buf[idxs]),
#     #     )
#
#     def sample_cur_batch_for_target(self, batch_size: int) -> Dict[str, tf.Tensor]:
#         idxs = np.arange(self.size - self.steps_per_task, self.size)
#
#         return dict(
#             obs=tf.convert_to_tensor(self.obs_buf[idxs]),
#             next_obs=tf.convert_to_tensor(self.next_obs_buf[idxs]),
#             actions=tf.convert_to_tensor(self.actions_buf[idxs]),
#             rewards=tf.convert_to_tensor(self.rewards_buf[idxs]),
#             done=tf.convert_to_tensor(self.done_buf[idxs]),
#         )
#
#     def sample_batch_for_bc(self, batch_size: int) -> Dict[str, tf.Tensor]:
#         cur_task_t = self.size % self.steps_per_task
#         idxs = np.random.randint(0, self.size - cur_task_t, size=batch_size)
#
#         return dict(
#             obs=tf.convert_to_tensor(self.obs_buf[idxs]),
#             actions=tf.convert_to_tensor(self.actions_buf[idxs]),
#             policy_mu=tf.convert_to_tensor(self.policy_mu_buf[idxs]),
#             policy_std=tf.convert_to_tensor(self.policy_std_buf[idxs]),
#             q1=tf.convert_to_tensor(self.q1_buf[idxs]),
#             q2=tf.convert_to_tensor(self.q2_buf[idxs]),
#         )

# MTR buffer
class MultiTimescaleReplayBuffer(ReplayBufferFIFO):
    def __init__(self, obs_dim=None, act_dim=None, size=1000000, num_buffers=20, beta=0.85, no_waste=True): 
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        print(size, num_buffers)
        self.num_buffers = num_buffers
        self._maxsize_per_buffer = size // num_buffers
        self._maxsize = num_buffers * self._maxsize_per_buffer
        self.beta = beta
        self.no_waste = no_waste
        self.count = 0
        
        if size % num_buffers != 0:
            print("Warning! Size is not divisible by number of buffers. New size is: ", self._maxsize)
        
        self.buffers = []
        for _ in range(num_buffers):
            self.buffers.append(ReplayBufferFIFO(self._maxsize_per_buffer))

        if no_waste:
            self.overflow_buffer = deque(maxlen=self._maxsize)

    def store(self, obs, action, reward, next_obs, done):
        self.add(obs, action, reward, next_obs, done)
    
    def sample_batch(self, batch_size):
        """SAC接口兼容方法，将sample方法的结果转换为字典格式，并确保数据类型一致"""
        obs, actions, rewards, next_obs, dones = self.sample(batch_size)
        
        obs = obs.astype(np.float32)
        actions = actions.astype(np.float32)
        rewards = rewards.astype(np.float32)
        next_obs = next_obs.astype(np.float32)
        dones = dones.astype(np.float32)
        
        return dict(
            obs=tf.convert_to_tensor(obs),
            next_obs=tf.convert_to_tensor(next_obs),
            actions=tf.convert_to_tensor(actions),
            rewards=tf.convert_to_tensor(rewards),
            done=tf.convert_to_tensor(dones),
        )

    def __len__(self):
        total_length = 0
        for buf in self.buffers:
            total_length += len(buf)
        if self.no_waste:
            total_length += len(self.overflow_buffer)
        return total_length

    @property
    def storage(self):
        """[(np.ndarray, float, float, np.ndarray, bool)]: content of the replay buffer"""
        total_storage = []
        for buf in self.buffers:
            total_storage.extend(buf.storage)
        return total_storage

    @property
    def buffer_size(self):
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples):
        """
        Check if n_samples samples can be sampled
        from the buffer.
        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self):
        """
        Check whether the replay buffer is full or not.
        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new transition to the buffer
        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        self.count += 1
        data = (obs_t, action, reward, obs_tp1, done)
        popped_data = self.buffers[0].add(*data)
        
        for i in range(1, self.num_buffers):
            #print("buffer ", i)
            #print("popped: ", popped_data)
            if popped_data == None:
                break
            if random.uniform(0, 1) < self.beta:
                popped_data = self.buffers[i].add(*popped_data)
            elif self.no_waste:
                self.overflow_buffer.appendleft(popped_data)
                break
            else:
                break
        if self.no_waste and (self.count > self._maxsize) and (len(self.overflow_buffer) != 0):
            self.overflow_buffer.pop()
            
    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        #storage = self.storage
        if self.no_waste:
            assert len(idxes) == (self.num_buffers + 1)
        else:
            assert len(idxes) == self.num_buffers
        for buf_idx in range(len(idxes)):
            for i in idxes[buf_idx]:
                #print(i)
                if buf_idx == 0 and self.no_waste:
                    data = self.overflow_buffer[i]
                else:
                    data = self.buffers[buf_idx - 1].storage[i]
                obs_t, action, reward, obs_tp1, done = data
                obses_t.append(np.array(obs_t, copy=False))
                actions.append(np.array(action, copy=False))
                rewards.append(reward)
                obses_tp1.append(np.array(obs_tp1, copy=False))
                dones.append(done)
            
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size, **_kwargs):
        """
        Sample a batch of experiences.
        :param batch_size: (int) How many transitions to sample.
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        all_idxes = []
        buf_lengths = [len(buf) for buf in self.buffers]
        if self.no_waste:
            buf_lengths.insert(0,len(self.overflow_buffer))

        buffer_batch_sizes = proportional(batch_size, buf_lengths)
        #print(buffer_batch_sizes)
        for i in range(len(buf_lengths)):
            idxes = [random.randint(0, buf_lengths[i] - 1) for _ in range(buffer_batch_sizes[i])]
            all_idxes.append(idxes)
        return self._encode_sample(all_idxes)

    def get_buffer_batch_sizes(self, batch_size):
        buf_lengths = [len(buf) for buf in self.buffers]
        if self.no_waste:
            buf_lengths.insert(0,len(self.overflow_buffer))

        return proportional(batch_size, buf_lengths)

def get_replay_buffer(name):
    if name == 'fifo':
        return ReplayBuffer
    elif name == 'reservoir':
        return ReservoirReplayBuffer
    elif name == 'multi_timescale':
        return MultiTimescaleReplayBuffer

if __name__=="__main__":
    import matplotlib.pyplot as plt
    
    buffer_type = 'multi_timescale'
    buffer_args = {'size': 100000, 'num_buffers': 20, 'beta': 0.85}
    reservoir = get_replay_buffer(buffer_type)(**buffer_args)
    lengths = []
    sim_length = 500000
    for i in range(sim_length):
        if i % 10000 == 1:
            print(i)
        reservoir.add(i, i, i, i, False)
        lengths.append(len(reservoir))
    sample = reservoir.sample(10000)
    if buffer_type=='multi_timescale':
        print(len(reservoir.overflow_buffer))
    times = []
    sample_times = []
    if buffer_type == 'reservoir':
        for data in reservoir.storage:
            times.append(sim_length-data[1][0])
    else:
        for data in reservoir.storage:
            times.append(sim_length-data[0])

    # Histogram of experience age
    fig=plt.figure()
    plt.hist(times, bins=range(0,sim_length,1000))
    plt.xlabel('Age', size=16)
    plt.ylabel('Number of experiences', size=16)
    plt.xlim([0, sim_length])
    plt.ylim([0, 1100])
    fig.savefig(buffer_type+"_replay_histogram")

    # Histogram of ages of sample
    fig2 = plt.figure()
    plt.hist(sample[0], bins=100)
    plt.xlabel('Age', size=16)
    plt.ylabel('Number of experiences', size=16)
    fig2.savefig(buffer_type+"_replay_histogram_sample")
    plt.show()