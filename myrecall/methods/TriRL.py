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
        
        # 初始化父类
        super(TriRL_SAC, self).__init__(**vanilla_sac_kwargs)
        
        # 初始化历史缓冲区
        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
        
        # 用于存储最近的历史数据
        self.recent_actions = np.zeros((self.history_length, self.action_dim))
        self.recent_rewards = np.zeros((self.history_length, 1))
        self.recent_observations = np.zeros((self.history_length, self.obs_dim))
    
    def get_action(self, o, deterministic=False):
        """获取动作，考虑历史信息
        
        Args:
            o: 当前观察
            deterministic: 是否使用确定性策略
            
        Returns:
            动作，额外信息
        """
        # 构建batch维度
        o = o.reshape(1, -1).astype(np.float32)
        
        # 准备历史数据
        prev_actions = self.recent_actions.reshape(1, self.history_length, self.action_dim).astype(np.float32)
        prev_rewards = self.recent_rewards.reshape(1, self.history_length, 1).astype(np.float32)
        prev_obs = self.recent_observations.reshape(1, self.history_length, self.obs_dim).astype(np.float32)
        
        history_data = (
            tf.convert_to_tensor(prev_actions),
            tf.convert_to_tensor(prev_rewards),
            tf.convert_to_tensor(prev_obs)
        )
        
        # 确定性策略
        if deterministic:
            mu, _, _, _ = self.actor(tf.convert_to_tensor(o), history_data)
            a = mu.numpy()[0]
        # 随机策略
        else:
            _, _, pi, _ = self.actor(tf.convert_to_tensor(o), history_data)
            a = pi.numpy()[0]
            
        # 更新历史记录
        self.recent_observations = np.roll(self.recent_observations, -1, axis=0)
        self.recent_observations[-1] = o[0]
        
        return a, {}
    
    def store_transition(self, o, a, r, o2, d):
        """存储转换并更新历史记录"""
        # 更新历史动作和奖励
        self.recent_actions = np.roll(self.recent_actions, -1, axis=0)
        self.recent_actions[-1] = a
        
        self.recent_rewards = np.roll(self.recent_rewards, -1, axis=0)
        self.recent_rewards[-1] = r
        
        # 调用父类方法存储到经验回放缓冲区
        super().store_transition(o, a, r, o2, d)
    
    def learn_on_batch(self, batch: Dict[str, tf.Tensor]) -> Dict:
        """使用批次数据进行学习，加入历史信息处理
        
        Args:
            batch: 包含'obs', 'actions', 'rewards', 'next_obs', 'done'的批次数据
            
        Returns:
            训练指标
        """
        # 由于经验回放中没有历史信息，这里我们为当前批次生成空的历史信息
        # 在实际应用中，你可能需要修改重放缓冲区来存储历史序列
        batch_size = tf.shape(batch['obs'])[0]
        
        # 创建空的历史信息
        empty_actions = tf.zeros((batch_size, self.history_length, self.action_dim))
        empty_rewards = tf.zeros((batch_size, self.history_length, 1))
        empty_obs = tf.zeros((batch_size, self.history_length, self.obs_dim))
        
        history_data = (empty_actions, empty_rewards, empty_obs)
        
        obs, next_obs = batch['obs'], batch['next_obs']
        actions, rewards = batch['actions'], batch['rewards']
        dones = batch['done']
        
        with tf.GradientTape(persistent=True) as g:
            # 计算critic损失
            _, _, pi, logp_pi = self.actor(next_obs, history_data)
            
            # Q值目标
            q1_pi_targ = self.target_critic1(next_obs, pi, history_data)
            q2_pi_targ = self.target_critic2(next_obs, pi, history_data)
            q_pi_targ = tf.minimum(q1_pi_targ, q2_pi_targ)
            
            # 获取当前任务的alpha
            seq_idx = tf.zeros(tf.shape(rewards)[0], dtype=tf.int32)
            alpha = self.get_alpha(seq_idx)
            
            # 计算目标值
            backup = rewards + self.gamma * (1 - dones) * (q_pi_targ - alpha * logp_pi)
            
            # 当前Q值
            q1 = self.critic1(obs, actions, history_data)
            q2 = self.critic2(obs, actions, history_data)
            
            # Critic损失
            q1_loss = 0.5 * tf.reduce_mean((q1 - backup) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q2 - backup) ** 2)
            critic_loss = q1_loss + q2_loss
            
            # Actor损失
            _, _, pi, logp_pi = self.actor(obs, history_data)
            q1_pi = self.critic1(obs, pi, history_data)
            q2_pi = self.critic2(obs, pi, history_data)
            q_pi = tf.minimum(q1_pi, q2_pi)
            
            actor_loss = tf.reduce_mean(alpha * logp_pi - q_pi)
            
            # Alpha损失（如果使用自动调整）
            if self.auto_alpha:
                target_entropy = -self.actor.action_space.shape[0]
                alpha_loss = -tf.reduce_mean(
                    self.all_log_alpha * tf.stop_gradient(logp_pi + target_entropy)
                )
            else:
                alpha_loss = 0.0
        
        # 计算梯度
        actor_gradients = g.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = g.gradient(critic_loss, self.critic_variables)
        
        if self.auto_alpha:
            alpha_gradient = g.gradient(alpha_loss, self.all_log_alpha)
        else:
            alpha_gradient = None
        del g
        
        # 应用梯度
        if self.clipnorm is not None:
            actor_gradients = [tf.clip_by_norm(g, self.clipnorm) for g in actor_gradients]
            critic_gradients = [tf.clip_by_norm(g, self.clipnorm) for g in critic_gradients]
        
        self.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.optimizer.apply_gradients(zip(critic_gradients, self.critic_variables))
        
        if self.auto_alpha:
            self.optimizer.apply_gradients([(alpha_gradient, self.all_log_alpha)])
        
        # Polyak平均更新目标网络
        for v, target_v in zip(
            self.critic1.trainable_variables, self.target_critic1.trainable_variables
        ):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)
        for v, target_v in zip(
            self.critic2.trainable_variables, self.target_critic2.trainable_variables
        ):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)
        
        # 返回指标
        return {
            "q1": tf.reduce_mean(q1),
            "q2": tf.reduce_mean(q2),
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "loss_pi": actor_loss,
            "logp_pi": tf.reduce_mean(logp_pi),
            "alpha": alpha,
            "loss_reg": tf.constant(0.0)
        }
    
    def on_episode_end(self):
        """在每个episode结束时重置历史记录"""
        self.recent_actions = np.zeros((self.history_length, self.action_dim))
        self.recent_rewards = np.zeros((self.history_length, 1))
        self.recent_observations = np.zeros((self.history_length, self.obs_dim))
    
    def on_task_start(self, current_task_idx: int) -> None:
        """在新任务开始时调用，重置历史记录"""
        super().on_task_start(current_task_idx)
        self.recent_actions = np.zeros((self.history_length, self.action_dim))
        self.recent_rewards = np.zeros((self.history_length, 1))
        self.recent_observations = np.zeros((self.history_length, self.obs_dim))
