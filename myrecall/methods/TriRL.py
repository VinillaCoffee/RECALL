import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

from myrecall.sac.sac import SAC
from myrecall.sac.models import RNNActor, RNNCritic


class TriRL_SAC(SAC):
    def __init__(
        self,
        history_length=15,
        context_dim=500,
        **vanilla_sac_kwargs
    ):
        """使用RNN网络进行历史状态编码的SAC实现
        
        Args:
            history_length: 历史序列长度
            context_dim: RNN隐藏状态维度
            **vanilla_sac_kwargs: 基础SAC的参数
        """
        # 保存RNN相关参数
        self.history_length = history_length
        self.context_dim = context_dim
        
        # 修改actor_cl和critic_cl为RNN版本
        vanilla_sac_kwargs["actor_cl"] = RNNActor
        vanilla_sac_kwargs["critic_cl"] = RNNCritic
        
        # 扩展actor_kwargs和critic_kwargs以包含RNN的参数
        actor_kwargs = vanilla_sac_kwargs.get("actor_kwargs", {})
        critic_kwargs = vanilla_sac_kwargs.get("critic_kwargs", {})
        
        # 添加RNN相关参数
        actor_kwargs.update({
            "history_length": history_length,
            "context_dim": context_dim,
            "obsr_dim": vanilla_sac_kwargs["env"].observation_space.shape[0],
        })
        
        critic_kwargs.update({
            "history_length": history_length,
            "context_dim": context_dim,
            "obsr_dim": vanilla_sac_kwargs["env"].observation_space.shape[0],
            "action_space": vanilla_sac_kwargs["env"].action_space,
        })
        
        # 更新kwargs
        vanilla_sac_kwargs["actor_kwargs"] = actor_kwargs
        vanilla_sac_kwargs["critic_kwargs"] = critic_kwargs
        
        # 保存critic_kwargs用于后续使用
        self.critic_kwargs = critic_kwargs
        
        # 初始化父类
        super(TriRL_SAC, self).__init__(**vanilla_sac_kwargs)
        
        # 初始化历史缓冲区
        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
        
        # 用于存储最近的历史数据
        self.recent_actions = np.zeros((self.history_length, self.action_dim))
        self.recent_rewards = np.zeros((self.history_length, 1))
        self.recent_observations = np.zeros((self.history_length, self.obs_dim))
        
        # 初始化张量版本的历史缓冲区，用于RNN输入
        self.h_actions = tf.zeros((1, self.history_length, self.action_dim), dtype=tf.float32)
        self.h_rewards = tf.zeros((1, self.history_length, 1), dtype=tf.float32)
        self.h_observations = tf.zeros((1, self.history_length, self.obs_dim), dtype=tf.float32)
        
        # 初始化RNN状态
        self.rnn_state = None
        
        # 重新定义critic_variables以包含RNN变量
        self.critic_variables = self.critic1.trainable_variables + self.critic2.trainable_variables
    
    @tf.function
    def get_action(self, o: tf.Tensor, deterministic: tf.Tensor = tf.constant(False)) -> tf.Tensor:
        """重写父类方法，添加历史信息处理"""
        # 准备历史数据
        history_data = (
            self.h_actions, 
            self.h_rewards, 
            self.h_observations
        )
        
        # 调用actor获取动作，传入历史数据
        mu, _, pi, _ = self.actor(tf.expand_dims(o, 0), history_data, training=False)
        
        # 根据是否确定性策略选择返回mu还是pi
        action = tf.cond(deterministic, lambda: mu[0], lambda: pi[0])
        
        # 确保动作维度正确
        if self.action_dim != tf.shape(action)[-1]:
            tf.print("警告: 动作维度不匹配! 期望:", self.action_dim, "得到:", tf.shape(action)[-1])
            # 如果维度不匹配，根据情况扩展或截断
            if tf.shape(action)[-1] < self.action_dim:
                # 如果维度太小，填充到正确维度
                pad_width = self.action_dim - tf.shape(action)[-1]
                action = tf.pad(action, [[0, pad_width]], constant_values=0.0)
            else:
                # 如果维度太大，截断到正确维度
                action = action[:self.action_dim]
        
        # 更新历史记录
        self.update_history(o, action, tf.constant(0.0, dtype=tf.float32))
        
        return action
    
    def on_episode_end(self):
        """在每个episode结束时重置历史记录"""
        self.recent_actions = np.zeros((self.history_length, self.action_dim))
        self.recent_rewards = np.zeros((self.history_length, 1))
        self.recent_observations = np.zeros((self.history_length, self.obs_dim))
        
        # 重置张量历史缓冲区
        self.h_actions = tf.zeros((1, self.history_length, self.action_dim), dtype=tf.float32)
        self.h_rewards = tf.zeros((1, self.history_length, 1), dtype=tf.float32)
        self.h_observations = tf.zeros((1, self.history_length, self.obs_dim), dtype=tf.float32)
        
        # 重置RNN状态
        self.rnn_state = None
    
    def on_task_start(self, current_task_idx: int) -> None:
        """在新任务开始时调用，重置历史记录"""
        super().on_task_start(current_task_idx)
        self.recent_actions = np.zeros((self.history_length, self.action_dim))
        self.recent_rewards = np.zeros((self.history_length, 1))
        self.recent_observations = np.zeros((self.history_length, self.obs_dim))
        
        # 重置张量历史缓冲区
        self.h_actions = tf.zeros((1, self.history_length, self.action_dim), dtype=tf.float32)
        self.h_rewards = tf.zeros((1, self.history_length, 1), dtype=tf.float32)
        self.h_observations = tf.zeros((1, self.history_length, self.obs_dim), dtype=tf.float32)
        
        # 重置RNN状态
        self.rnn_state = None

    def update_history(self, o, a, r):
        """更新历史信息
        
        Args:
            o: 观测值
            a: 动作
            r: 奖励
        """
        # 将输入转换为张量，保持原来的数据类型
        o = tf.convert_to_tensor(o)
        if len(o.shape) == 1:
            o = tf.expand_dims(o, axis=0)  # 增加批次维度
        
        a = tf.convert_to_tensor(a)
        if len(a.shape) == 1:
            a = tf.expand_dims(a, axis=0)  # 增加批次维度
            
        # 确保动作维度正确
        if tf.shape(a)[-1] != self.action_dim:
            tf.print("警告: 在update_history中动作维度不匹配! 期望:", self.action_dim, "得到:", tf.shape(a)[-1])
            # 如果维度不匹配，根据情况扩展或截断
            if tf.shape(a)[-1] < self.action_dim:
                # 如果维度太小，填充到正确维度
                pad_width = self.action_dim - tf.shape(a)[-1]
                a = tf.pad(a, [[0, 0], [0, pad_width]], constant_values=0.0)
            else:
                # 如果维度太大，截断到正确维度
                a = a[:, :self.action_dim]
        
        r = tf.convert_to_tensor(r)
        if len(r.shape) == 0:
            r = tf.reshape(r, (1, 1))  # 增加批次和特征维度
        elif len(r.shape) == 1:
            r = tf.expand_dims(r, axis=-1)  # 增加特征维度
        
        # 更新numpy历史记录 - 转换为numpy时进行数据类型转换
        self.recent_actions = np.roll(self.recent_actions, -1, axis=0)
        self.recent_actions[-1] = a.numpy()[0]
        
        self.recent_rewards = np.roll(self.recent_rewards, -1, axis=0)
        self.recent_rewards[-1] = r.numpy()[0]
        
        self.recent_observations = np.roll(self.recent_observations, -1, axis=0)
        self.recent_observations[-1] = o.numpy()[0]
        
        # 更新历史序列张量 - 确保历史张量的数据类型与输入一致
        self.h_observations = tf.concat([
            tf.cast(self.h_observations[:, 1:], dtype=o.dtype), 
            tf.expand_dims(o, axis=1)
        ], axis=1)
        
        self.h_actions = tf.concat([
            tf.cast(self.h_actions[:, 1:], dtype=a.dtype), 
            tf.expand_dims(a, axis=1)
        ], axis=1)
        
        self.h_rewards = tf.concat([
            tf.cast(self.h_rewards[:, 1:], dtype=r.dtype), 
            tf.expand_dims(r, axis=1)
        ], axis=1)
        
        # 更新RNN状态
        if hasattr(self, 'rnn_state') and self.rnn_state is not None:
            # 收集历史数据
            history_data = (self.h_actions, self.h_rewards, self.h_observations)
            # 调用actor以更新RNN状态，设置training=False
            _, _, _, self.rnn_state = self.actor(o, history_data, training=False)
    
    def store_transition(self, o, a, r, o2, d):
        """存储转换并更新历史记录"""
        # 更新历史数据
        self.update_history(o, a, r)
        
        # 调用父类方法存储到经验回放缓冲区
        super().store_transition(o, a, r, o2, d)
    
    def get_gradients(
        self,
        seq_idx: tf.Tensor,
        aux_batch: Dict[str, tf.Tensor],
        obs: tf.Tensor,
        next_obs: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        done: tf.Tensor,
    ) -> Tuple[Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]], Dict]:
        """重写get_gradients方法处理历史数据"""
        # 从缓冲区获取历史数据
        batch_size = tf.shape(obs)[0]
        h_a = tf.zeros((batch_size, self.history_length, self.action_dim), dtype=tf.float32)
        h_r = tf.zeros((batch_size, self.history_length, 1), dtype=tf.float32)
        h_o = tf.zeros((batch_size, self.history_length, self.obs_dim), dtype=tf.float32)
        
        # 确保actions、rewards和obs有正确的形状以用于拼接
        if len(tf.shape(actions)) == 2:
            actions_3d = tf.expand_dims(actions, axis=1)  # [batch_size, 1, action_dim]
        else:
            actions_3d = actions
            
        # 处理rewards，确保其具有正确的维度 [batch_size, 1, 1]
        if len(tf.shape(rewards)) == 1:
            rewards_3d = tf.reshape(rewards, [batch_size, 1, 1])
        elif len(tf.shape(rewards)) == 2 and tf.shape(rewards)[1] == 1:
            rewards_3d = tf.expand_dims(rewards, axis=1)
        else:
            rewards_3d = rewards
            
        # 处理done，确保其具有正确的维度
        if len(tf.shape(done)) == 1:
            done = tf.expand_dims(done, axis=-1)
            
        # 处理obs，确保其具有正确的维度 [batch_size, 1, obs_dim]
        if len(tf.shape(obs)) == 2:
            obs_3d = tf.expand_dims(obs, axis=1)
        else:
            obs_3d = obs
        
        # 构建历史数据
        history_data = (h_a, h_r, h_o)
        next_history_data = (
            tf.concat([h_a[:, 1:], actions_3d], axis=1),
            tf.concat([h_r[:, 1:], rewards_3d], axis=1),
            tf.concat([h_o[:, 1:], obs_3d], axis=1)
        )
        
        with tf.GradientTape(persistent=True) as g:
            if self.auto_alpha:
                log_alpha = self.get_log_alpha(obs)
            else:
                log_alpha = tf.math.log(self.alpha)

            # Main outputs from computation graph with history data
            mu, log_std, pi, logp_pi = self.actor(obs, history_data, training=True)
            # critic返回q值和其他信息，只取第一个元素
            q1 = self.critic1(obs, actions, pre_act_rew=history_data, training=True)[0]
            q2 = self.critic2(obs, actions, pre_act_rew=history_data, training=True)[0]

            # compose q with pi, for pi-learning
            q1_pi = self.critic1(obs, pi, pre_act_rew=history_data, training=True)[0]
            q2_pi = self.critic2(obs, pi, pre_act_rew=history_data, training=True)[0]

            # get actions and log probs of actions for next states, for Q-learning
            _, _, pi_next, logp_pi_next = self.actor(next_obs, next_history_data, training=True)

            # target q values, using actions from *current* policy
            target_q1 = self.target_critic1(next_obs, pi_next, pre_act_rew=next_history_data, training=True)[0]
            target_q2 = self.target_critic2(next_obs, pi_next, pre_act_rew=next_history_data, training=True)[0]

            # Min Double-Q:
            min_q_pi = tf.minimum(q1_pi, q2_pi)
            min_target_q = tf.minimum(target_q1, target_q2)

            # Entropy-regularized Bellman backup for Q functions, using Clipped Double-Q targets
            q_backup = tf.stop_gradient(
                rewards + self.gamma * (1 - done) * (min_target_q - tf.math.exp(log_alpha) * logp_pi_next)
            )

            # Soft actor-critic losses
            pi_loss = tf.reduce_mean(tf.math.exp(log_alpha) * logp_pi - min_q_pi)
            q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
            value_loss = q1_loss + q2_loss

            auxiliary_loss = self.get_auxiliary_loss(seq_idx, aux_batch)
            metrics = dict(
                pi_loss=pi_loss,
                q1_loss=q1_loss,
                q2_loss=q2_loss,
                q1=q1,
                q2=q2,
                logp_pi=logp_pi,
                reg_loss=auxiliary_loss,
            )

            pi_loss += auxiliary_loss
            value_loss += auxiliary_loss

            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(
                    log_alpha * tf.stop_gradient(logp_pi + self.target_entropy)
                )

        # Compute gradients
        actor_gradients = g.gradient(pi_loss, self.actor.trainable_variables)
        critic_gradients = g.gradient(value_loss, self.critic_variables)
        if self.auto_alpha:
            alpha_gradient = g.gradient(alpha_loss, self.all_log_alpha)
        else:
            alpha_gradient = None
        del g

        gradients = (actor_gradients, critic_gradients, alpha_gradient)
        return gradients, metrics
