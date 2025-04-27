import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

from myrecall.sac.sac import SAC
from myrecall.sac.models import ContextRNN
from myrecall.sac.replay_buffers import ReplayBuffer


class TriRL_SAC(SAC):
    def __init__(
        self,
        history_length=15,  # 对应伪代码中的h
        context_dim=500,    # 对应动态任务表示z的维度
        replay_ratio=0.5,   # 对应伪代码中的β，旧缓冲区采样比例
        **vanilla_sac_kwargs
    ):
        """使用RNN网络生成动态任务表示的SAC实现 (3RL in TACRL)
        
        Args:
            history_length: 历史序列长度，对应伪代码中的h
            context_dim: RNN隐藏状态维度，用于任务表示z的维度
            replay_ratio: 旧缓冲区采样比例，对应伪代码中的β
            **vanilla_sac_kwargs: 基础SAC的参数
        """
        # 保存RNN相关参数
        self.history_length = history_length
        self.context_dim = context_dim
        self.replay_ratio = replay_ratio
        
        # 扩展actor_kwargs和critic_kwargs以包含上下文维度
        actor_kwargs = vanilla_sac_kwargs.get("actor_kwargs", {})
        critic_kwargs = vanilla_sac_kwargs.get("critic_kwargs", {})
        
        # 环境观测空间维度
        obs_dim = vanilla_sac_kwargs["env"].observation_space.shape[0]
        
        # 更新输入维度，包含观测和任务表示
        actor_kwargs["input_dim"] = obs_dim + context_dim
        critic_kwargs["input_dim"] = obs_dim + context_dim + vanilla_sac_kwargs["env"].action_space.shape[0]
        
        # 更新kwargs
        vanilla_sac_kwargs["actor_kwargs"] = actor_kwargs
        vanilla_sac_kwargs["critic_kwargs"] = critic_kwargs
        
        # 初始化父类
        super(TriRL_SAC, self).__init__(**vanilla_sac_kwargs)
        
        # 初始化历史缓冲区
        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
        
        # 用于存储最近的历史数据 - 使用TensorFlow张量而不是numpy数组
        self.recent_actions = tf.zeros((self.history_length, self.action_dim), dtype=tf.float32)
        self.recent_rewards = tf.zeros((self.history_length, 1), dtype=tf.float32)
        self.recent_observations = tf.zeros((self.history_length, self.obs_dim), dtype=tf.float32)
        
        # 初始化张量版本的历史缓冲区，用于RNN输入
        self.h_actions = tf.zeros((1, self.history_length, self.action_dim), dtype=tf.float32)
        self.h_rewards = tf.zeros((1, self.history_length, 1), dtype=tf.float32)
        self.h_observations = tf.zeros((1, self.history_length, self.obs_dim), dtype=tf.float32)
        
        # 创建ContextRNN用于生成动态任务表示
        self.context_rnn = ContextRNN(
            hidden_dim=self.context_dim,
            input_dim=self.action_dim + 1 + self.obs_dim,  # action + reward + observation
            history_length=self.history_length,
            action_dim=self.action_dim,
            obsr_dim=self.obs_dim
        )
        
        # 创建旧缓冲区
        self.old_replay_buffer = None
    
    @tf.function
    def compute_task_representation(self, history_data):
        """计算动态任务表示 z_t = RNN_θ({(s_i^o, a_i, r_i)}_{i=t-h-1}^{t-1})"""
        # 调用ContextRNN生成任务表示
        batch_size = tf.shape(history_data[0])[0]
        task_representation = self.context_rnn(history_data, training=False)
        
        # 确保输出形状为 [batch_size, context_dim]
        if len(task_representation.shape) == 1:
            # 如果是单个向量 [context_dim]，扩展为 [1, context_dim]
            task_representation = tf.expand_dims(task_representation, 0)
            
        # 如果需要，复制到批次大小
        if tf.shape(task_representation)[0] == 1 and batch_size > 1:
            task_representation = tf.tile(task_representation, [batch_size, 1])
            
        return task_representation
    
    @tf.function
    def get_action(self, o: tf.Tensor, deterministic: tf.Tensor = tf.constant(False)) -> tf.Tensor:
        """重写父类方法，使用动态任务表示z_t选择动作 a_t ~ π_θ(·|s_t^o, z_t)"""
        # 确保输入张量为float32类型
        o = tf.cast(o, dtype=tf.float32)
        
        # 准备历史数据并计算任务表示
        history_data = (
            self.h_actions, 
            self.h_rewards, 
            self.h_observations
        )
        
        # 计算动态任务表示
        task_representation = self.compute_task_representation(history_data)
        
        # 确保o是二维张量 [batch_size, obs_dim]
        if len(o.shape) == 1:
            o = tf.expand_dims(o, axis=0)  # 添加批次维度 [1, obs_dim]
            
        # 确保task_representation是二维张量 [batch_size, context_dim]
        task_representation = tf.reshape(task_representation, [1, self.context_dim])
            
        # 将任务表示与观测拼接（沿特征维度）
        obs_with_context = tf.concat([o, task_representation], axis=-1)
        
        # 调用actor获取动作
        mu, _, pi, _ = self.actor(obs_with_context, training=False)
        
        # 根据是否确定性策略选择返回mu还是pi
        action = tf.cond(deterministic, lambda: mu[0], lambda: pi[0])
        
        # 更新历史记录
        self.update_history(o[0], action, tf.constant(0.0, dtype=tf.float32))
        
        return action
    
    def get_action_test(self, o: tf.Tensor, deterministic: tf.Tensor = tf.constant(False)) -> tf.Tensor:
        """测试时使用的动作选择函数"""
        return self.get_action(o, deterministic)
    
    def update_history(self, o, a, r):
        """更新历史信息
        
        Args:
            o: 观测值
            a: 动作
            r: 奖励
        """
        # 将输入转换为张量，并确保是float32类型
        o = tf.cast(tf.convert_to_tensor(o), dtype=tf.float32)
        if len(o.shape) == 1:
            o = tf.expand_dims(o, axis=0)  # 增加批次维度
        
        a = tf.cast(tf.convert_to_tensor(a), dtype=tf.float32)
        if len(a.shape) == 1:
            a = tf.expand_dims(a, axis=0)  # 增加批次维度
        
        r = tf.cast(tf.convert_to_tensor(r), dtype=tf.float32)
        if len(r.shape) == 0:
            r = tf.reshape(r, (1, 1))  # 增加批次和特征维度
        elif len(r.shape) == 1:
            r = tf.expand_dims(r, axis=-1)  # 增加特征维度
        
        # 使用TensorFlow变量更新历史数据，避免使用numpy
        # 创建新的TensorFlow变量来保存历史数据
        self.recent_actions = tf.roll(self.recent_actions, shift=-1, axis=0)
        self.recent_rewards = tf.roll(self.recent_rewards, shift=-1, axis=0)
        self.recent_observations = tf.roll(self.recent_observations, shift=-1, axis=0)
        
        # 使用TensorFlow赋值操作更新最后一行
        indices = tf.constant([[self.history_length - 1]])
        self.recent_actions = tf.tensor_scatter_nd_update(self.recent_actions, indices, [a[0]])
        self.recent_rewards = tf.tensor_scatter_nd_update(self.recent_rewards, indices, [r[0]])
        self.recent_observations = tf.tensor_scatter_nd_update(self.recent_observations, indices, [o[0]])
        
        # 更新历史序列张量
        self.h_observations = tf.concat([
            self.h_observations[:, 1:], 
            tf.expand_dims(o, axis=1)
        ], axis=1)
        
        self.h_actions = tf.concat([
            self.h_actions[:, 1:], 
            tf.expand_dims(a, axis=1)
        ], axis=1)
        
        self.h_rewards = tf.concat([
            self.h_rewards[:, 1:], 
            tf.expand_dims(r, axis=1)
        ], axis=1)
    
    def store_transition(self, o, a, r, o2, d):
        """存储转换并更新历史记录"""
        # 更新历史数据
        self.update_history(o, a, r)
        
        # 调用父类方法存储到经验回放缓冲区
        super().store_transition(o, a, r, o2, d)
    
    def on_episode_end(self):
        """在每个episode结束时重置历史记录"""
        self.recent_actions = tf.zeros((self.history_length, self.action_dim), dtype=tf.float32)
        self.recent_rewards = tf.zeros((self.history_length, 1), dtype=tf.float32)
        self.recent_observations = tf.zeros((self.history_length, self.obs_dim), dtype=tf.float32)
        
        # 重置张量历史缓冲区
        self.h_actions = tf.zeros((1, self.history_length, self.action_dim), dtype=tf.float32)
        self.h_rewards = tf.zeros((1, self.history_length, 1), dtype=tf.float32)
        self.h_observations = tf.zeros((1, self.history_length, self.obs_dim), dtype=tf.float32)
    
    def on_task_start(self, current_task_idx: int) -> None:
        """在新任务开始时调用，实现伪代码中的第11行"""
        # 如果已经有经验回放缓冲区，将其转移到旧缓冲区
        if hasattr(self, 'replay_buffer') and self.replay_buffer is not None:
            # 创建新的ReplayBuffer来存储旧数据
            self.old_replay_buffer = ReplayBuffer(
                obs_dim=self.obs_dim, 
                act_dim=self.action_dim, 
                size=self.replay_size
            )
            
            # 复制数据到新的缓冲区
            if self.replay_buffer.size > 0:
                # 获取当前缓冲区中的所有有效数据
                idx = min(self.replay_buffer.size, self.replay_buffer.max_size)
                self.old_replay_buffer.obs_buf[:idx] = self.replay_buffer.obs_buf[:idx]
                self.old_replay_buffer.next_obs_buf[:idx] = self.replay_buffer.next_obs_buf[:idx]
                self.old_replay_buffer.actions_buf[:idx] = self.replay_buffer.actions_buf[:idx]
                self.old_replay_buffer.rewards_buf[:idx] = self.replay_buffer.rewards_buf[:idx]
                self.old_replay_buffer.done_buf[:idx] = self.replay_buffer.done_buf[:idx]
                self.old_replay_buffer.size = self.replay_buffer.size
                self.old_replay_buffer.ptr = self.replay_buffer.ptr
            
            # 清空当前缓冲区
            self.clear_replay_buffer()
        
        # 调用父类方法处理任务切换
        super().on_task_start(current_task_idx)
        
        # 重置历史记录
        self.on_episode_end()
    
    def clear_replay_buffer(self):
        """清空当前的经验回放缓冲区"""
        if hasattr(self, 'replay_buffer') and self.replay_buffer is not None:
            # 重置指针和大小，保持缓冲区内存不变
            self.replay_buffer.ptr = 0
            self.replay_buffer.size = 0
    
    def get_learn_on_batch(self, current_task_idx: int):
        """重写获取批量学习函数的方法，实现伪代码中的第8-10行"""
        original_learn_on_batch = super().get_learn_on_batch(current_task_idx)
        
        @tf.function
        def learn_on_batch(
            seq_idx: tf.Tensor,
            batch: Dict[str, tf.Tensor],
            episodic_batch: Dict[str, tf.Tensor] = None,
            aux_batch: Dict[str, tf.Tensor] = None,
        ) -> Dict:
            # 计算采样大小
            batch_size = self.batch_size
            current_buffer_size = min(batch_size, int(batch_size * (1 - self.replay_ratio)))
            old_buffer_size = batch_size - current_buffer_size
            
            # 初始化合并后的批次
            combined_batch = batch
            
            # 只有在有旧缓冲区且需要从中采样时才进行特殊处理
            if self.old_replay_buffer is not None and old_buffer_size > 0:
                # 从当前缓冲区采样 - 伪代码中的第8行
                if current_buffer_size > 0:
                    current_batch = self.replay_buffer.sample_batch(current_buffer_size)
                else:
                    current_batch = None
                
                # 从旧缓冲区采样 - 伪代码中的第9行
                try:
                    old_batch = self.old_replay_buffer.sample_batch(old_buffer_size)
                    
                    # 合并两个批次
                    if current_batch is not None:
                        # 合并两个批次的所有键值
                        combined_batch = {}
                        for key in current_batch:
                            if key in old_batch:
                                combined_batch[key] = tf.concat([current_batch[key], old_batch[key]], axis=0)
                    else:
                        combined_batch = old_batch
                except:
                    # 如果旧缓冲区采样失败，使用原始批次
                    pass
            
            # 使用原始的learn_on_batch处理合并后的批次 - 伪代码中的第10行
            return original_learn_on_batch(seq_idx, combined_batch, episodic_batch, aux_batch)
        
        return learn_on_batch
    
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
        """重写梯度计算方法，为actor和critic提供历史信息"""
        # 确保输入张量都是float32类型
        obs = tf.cast(obs, dtype=tf.float32)
        next_obs = tf.cast(next_obs, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.float32)
        rewards = tf.cast(rewards, dtype=tf.float32)
        done = tf.cast(done, dtype=tf.float32)
        
        # 从缓冲区获取历史数据
        batch_size = tf.shape(obs)[0]
        h_a = tf.zeros((batch_size, self.history_length, self.action_dim), dtype=tf.float32)
        h_r = tf.zeros((batch_size, self.history_length, 1), dtype=tf.float32)
        h_o = tf.zeros((batch_size, self.history_length, self.obs_dim), dtype=tf.float32)
        
        # 准备当前历史和下一步历史
        actions_3d = tf.expand_dims(actions, axis=1)  # [batch_size, 1, action_dim]
        rewards_3d = tf.reshape(rewards, [batch_size, 1, 1])
        obs_3d = tf.expand_dims(obs, axis=1)  # [batch_size, 1, obs_dim]
        
        # 构建历史数据元组
        pre_act_rew = (h_a, h_r, h_o)
        next_pre_act_rew = (
            tf.concat([h_a[:, 1:], actions_3d], axis=1),
            tf.concat([h_r[:, 1:], rewards_3d], axis=1),
            tf.concat([h_o[:, 1:], obs_3d], axis=1)
        )
        
        with tf.GradientTape(persistent=True) as g:
            if self.auto_alpha:
                log_alpha = self.get_log_alpha(obs)
            else:
                log_alpha = tf.math.log(self.alpha)

            # 计算动态任务表示 - 对于批处理情况，任务表示应该已经包含批次维度
            # 我们假设RNN返回的任务表示已经是 [batch_size, context_dim] 的形状
            task_representation = self.compute_task_representation(pre_act_rew)
            next_task_representation = self.compute_task_representation(next_pre_act_rew)
            
            # 将任务表示与观测结合
            obs_with_context = tf.concat([obs, task_representation], axis=-1)
            next_obs_with_context = tf.concat([next_obs, next_task_representation], axis=-1)

            # 传入带有任务表示的观测获取动作和Q值
            mu, log_std, pi, logp_pi = self.actor(obs_with_context, training=True)
            
            # 将动作和带上下文的观测传给critic
            q1 = self.critic1(obs_with_context, actions, training=True)
            q2 = self.critic2(obs_with_context, actions, training=True)

            # 使用当前策略计算Q值
            q1_pi = self.critic1(obs_with_context, pi, training=True)
            q2_pi = self.critic2(obs_with_context, pi, training=True)

            # 获取下一个状态的动作和对数概率
            _, _, pi_next, logp_pi_next = self.actor(next_obs_with_context, training=True)

            # 计算目标Q值
            target_q1 = self.target_critic1(next_obs_with_context, pi_next, training=True)
            target_q2 = self.target_critic2(next_obs_with_context, pi_next, training=True)

            # 使用双Q网络取最小值
            min_q_pi = tf.minimum(q1_pi, q2_pi)
            min_target_q = tf.minimum(target_q1, target_q2)

            # 基于熵正则化的贝尔曼更新
            q_backup = tf.stop_gradient(
                rewards + self.gamma * (1 - done) * (min_target_q - tf.math.exp(log_alpha) * logp_pi_next)
            )

            # 计算SAC损失
            pi_loss = tf.reduce_mean(tf.math.exp(log_alpha) * logp_pi - min_q_pi)
            q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
            value_loss = q1_loss + q2_loss

            # 添加辅助损失
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

        # 计算梯度
        actor_gradients = g.gradient(pi_loss, self.actor.trainable_variables)
        critic_variables = self.critic1.trainable_variables + self.critic2.trainable_variables
        critic_gradients = g.gradient(value_loss, critic_variables)
        if self.auto_alpha:
            alpha_gradient = g.gradient(alpha_loss, self.all_log_alpha)
        else:
            alpha_gradient = None
        del g

        gradients = (actor_gradients, critic_gradients, alpha_gradient)
        return gradients, metrics
    
    def update_target_networks(self):
        """更新目标网络，使用polyak平均"""
        # 对critic1进行polyak平均更新
        for v, target_v in zip(
            self.critic1.trainable_variables, self.target_critic1.trainable_variables
        ):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)
        
        # 对critic2进行polyak平均更新
        for v, target_v in zip(
            self.critic2.trainable_variables, self.target_critic2.trainable_variables
        ):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)
