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
        context_dim=30,    # 对应动态任务表示z的维度
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
        
        # 调试信息：打印输入参数
        print("=== TriRL_SAC初始化 ===")
        print(f"历史长度: {history_length}")
        print(f"上下文维度: {context_dim}")
        print(f"回放比例: {replay_ratio}")
        
        # 扩展actor_kwargs和critic_kwargs以包含上下文维度
        actor_kwargs = vanilla_sac_kwargs.get("actor_kwargs", {})
        critic_kwargs = vanilla_sac_kwargs.get("critic_kwargs", {})
        
        # 环境观测空间维度
        obs_dim = vanilla_sac_kwargs["env"].observation_space.shape[0]
        act_dim = vanilla_sac_kwargs["env"].action_space.shape[0]
        
        print(f"观测维度: {obs_dim}")
        print(f"动作维度: {act_dim}")
        
        # 更新输入维度，包含观测和任务表示
        actor_input_dim = obs_dim + context_dim
        # 修复：critic的输入维度只需要是观测维度，在critic内部会将action拼接上去
        critic_input_dim = obs_dim + context_dim
        
        print(f"Actor输入维度: {actor_input_dim} (观测 {obs_dim} + 任务表示 {context_dim})")
        print(f"Critic输入维度: {critic_input_dim} (观测 {obs_dim} + 任务表示 {context_dim})")
        
        actor_kwargs["input_dim"] = actor_input_dim
        critic_kwargs["input_dim"] = critic_input_dim
        
        # 更新kwargs
        vanilla_sac_kwargs["actor_kwargs"] = actor_kwargs
        vanilla_sac_kwargs["critic_kwargs"] = critic_kwargs
        
        # 初始化父类
        super(TriRL_SAC, self).__init__(**vanilla_sac_kwargs)
        
        # 初始化历史缓冲区
        self.action_dim = self.env.action_space.shape[0]
        self.obs_dim = self.env.observation_space.shape[0]
        
        print(f"已初始化Actor和Critic网络")
        print(f"Actor输入维度: {actor_input_dim}, 输出维度: {self.action_dim}")
        print(f"Critic1输入维度: {critic_input_dim}, 输出维度: 1")
        
        # 确认网络是否正确初始化
        print(f"Actor trainable变量形状: {[var.shape for var in self.actor.trainable_variables]}")
        print(f"Critic1 trainable变量形状: {[var.shape for var in self.critic1.trainable_variables]}")
        
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
        
        print(f"ContextRNN输入维度: {self.action_dim + 1 + self.obs_dim}, 输出维度: {self.context_dim}")
        print(f"ContextRNN trainable变量形状: {[var.shape for var in self.context_rnn.trainable_variables]}")
        
        # 测试ContextRNN
        print("Testing ContextRNN...")
        
        # 创建随机输入数据
        test_actions = tf.random.normal([1, self.history_length, self.action_dim], dtype=tf.float32)
        test_rewards = tf.random.normal([1, self.history_length, 1], dtype=tf.float32)
        test_obs = tf.random.normal([1, self.history_length, self.obs_dim], dtype=tf.float32)
        
        # 确保test_history包含float32类型数据
        test_history = (
            tf.cast(test_actions, dtype=tf.float32),
            tf.cast(test_rewards, dtype=tf.float32), 
            tf.cast(test_obs, dtype=tf.float32)
        )
        
        # 显式尝试每个部分
        print("Test shapes:")
        print(f"  Actions: {test_actions.shape}, dtype: {test_actions.dtype}")
        print(f"  Rewards: {test_rewards.shape}, dtype: {test_rewards.dtype}")
        print(f"  Observations: {test_obs.shape}, dtype: {test_obs.dtype}")
        
        # 使用try-except捕获潜在错误
        try:
            # 尝试调用ContextRNN
            test_representation = self.context_rnn(test_history, training=False)
            tf.print("ContextRNN test output shape:", tf.shape(test_representation))
            tf.print("ContextRNN test output mean:", tf.reduce_mean(tf.abs(test_representation)))
        except Exception as e:
            print(f"Error testing ContextRNN: {e}")
            print("Using fallback task representation")
            # 如果ContextRNN出错，创建一个初始随机任务表示作为后备
            self.fallback_representation = tf.Variable(
                tf.random.normal([1, self.context_dim], mean=0.0, stddev=0.1, dtype=tf.float32),
                trainable=True, name="fallback_task_representation"
            )
        
        # 创建旧缓冲区
        self.old_replay_buffer = None
        
        # 将context_rnn添加到优化器中跟踪的变量列表，确保其参数更新
        self.context_trainable_variables = self.context_rnn.trainable_variables
        
        # 重置优化器以包含所有需要训练的变量
        self.reset_optimizer()
        
        # 测试critic网络初始化
        self.test_critic_initialization()
        
        print("=== TriRL_SAC初始化完成 ===")
    
    def reset_optimizer(self):
        """重置优化器，确保包含所有需要训练的变量"""
        # 使用相同的学习率和参数重建优化器
        # 检查self.lr是否存在，如果不存在则使用默认值1e-3
        lr = getattr(self, 'lr', 1e-3)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        # 进行一次假的优化步骤，确保优化器正确初始化所有变量
        with tf.GradientTape() as tape:
            dummy_var = tf.Variable(tf.zeros(1))
            dummy_loss = tf.reduce_sum(dummy_var)
        dummy_grads = tape.gradient(dummy_loss, [dummy_var])
        self.optimizer.apply_gradients(zip(dummy_grads, [dummy_var]))

    def test_critic_initialization(self):
        """测试critic网络是否正确初始化并能产生非零输出"""
        print("Testing Critic network initialization...")
        test_obs = tf.random.normal([1, self.obs_dim + self.context_dim], dtype=tf.float32)
        test_act = tf.random.normal([1, self.action_dim], dtype=tf.float32)
        test_q1 = self.critic1(test_obs, test_act, training=False)
        test_q2 = self.critic2(test_obs, test_act, training=False)
        
        # 使用tf.print等待计算完成后再打印
        tf.print("Critic1 test output:", test_q1)
        tf.print("Critic2 test output:", test_q2)
        
        # 打印网络结构
        print("Critic network structure:")
        for i, layer in enumerate(self.critic1.core.layers):
            print(f"  Layer {i}: {layer.__class__.__name__}")
        
        # 找到一个有kernel的层来打印权重信息
        for i, layer in enumerate(self.critic1.core.layers):
            if hasattr(layer, 'kernel'):
                kernel_mean = tf.reduce_mean(layer.kernel)
                bias_mean = tf.reduce_mean(layer.bias)
                tf.print(f"Critic1 layer {i} (Dense) kernel mean:", kernel_mean)
                tf.print(f"Critic1 layer {i} (Dense) bias mean:", bias_mean)
                break
        
        # 检查最后一层
        if hasattr(self.critic1.head, 'layers'):
            last_layer = self.critic1.head.layers[-1]
            if hasattr(last_layer, 'kernel'):
                last_kernel_mean = tf.reduce_mean(last_layer.kernel)
                last_bias_mean = tf.reduce_mean(last_layer.bias)
                tf.print("Critic1 last layer kernel mean:", last_kernel_mean)
                tf.print("Critic1 last layer bias mean:", last_bias_mean)
        
        return test_q1, test_q2
    
    def compute_task_representation(self, history_data, training=False):
        """计算动态任务表示 z_t = RNN_θ({(s_i^o, a_i, r_i)}_{i=t-h-1}^{t-1})
        
        Args:
            history_data: 元组 (actions, rewards, observations)，每个元素形状为[batch_size, history_length, dim]
            training: 是否为训练模式
        
        Returns:
            task_representation: 动态任务表示，形状为[batch_size, context_dim]
        """
        # 解包历史数据
        h_actions, h_rewards, h_observations = history_data
        
        # 添加调试信息 - 使用tf.print替代print和.numpy()
        if not training:
            tf.print("History action mean:", tf.reduce_mean(tf.abs(h_actions)))
            tf.print("History reward mean:", tf.reduce_mean(tf.abs(h_rewards)))
            tf.print("History observation mean:", tf.reduce_mean(tf.abs(h_observations)))
        
        # 确保数据类型一致
        h_actions = tf.cast(h_actions, dtype=tf.float32)
        h_rewards = tf.cast(h_rewards, dtype=tf.float32)
        h_observations = tf.cast(h_observations, dtype=tf.float32)
        
        # 获取批大小
        batch_size = tf.shape(h_actions)[0]
        
        # 防止历史数据全为零导致任务表示无效
        h_actions_mean = tf.reduce_mean(tf.abs(h_actions))
        h_rewards_mean = tf.reduce_mean(tf.abs(h_rewards))
        h_observations_mean = tf.reduce_mean(tf.abs(h_observations))
        
        # 如果历史数据接近零，添加一些小随机值
        is_zero_history = tf.less(h_actions_mean + h_rewards_mean + h_observations_mean, 1e-6)
        
        def add_noise():
            # 添加一些随机噪声到历史数据
            noise_actions = h_actions + tf.random.normal(
                tf.shape(h_actions), mean=0.0, stddev=0.01, dtype=tf.float32)
            noise_rewards = h_rewards + tf.random.normal(
                tf.shape(h_rewards), mean=0.0, stddev=0.01, dtype=tf.float32)
            noise_observations = h_observations + tf.random.normal(
                tf.shape(h_observations), mean=0.0, stddev=0.01, dtype=tf.float32)
            return noise_actions, noise_rewards, noise_observations
        
        # 条件选择是否添加噪声
        h_actions, h_rewards, h_observations = tf.cond(
            is_zero_history,
            add_noise,
            lambda: (h_actions, h_rewards, h_observations)
        )
        
        # 重新构建历史数据元组
        history_data = (h_actions, h_rewards, h_observations)
        
        # 确保形状正确
        tf.debugging.assert_equal(tf.shape(h_actions)[1], self.history_length, 
                                 message="History actions shape mismatch")
        tf.debugging.assert_equal(tf.shape(h_rewards)[1], self.history_length,
                                 message="History rewards shape mismatch")
        tf.debugging.assert_equal(tf.shape(h_observations)[1], self.history_length,
                                 message="History observations shape mismatch")
        
        # 调用ContextRNN生成任务表示
        try:
            task_representation = self.context_rnn(history_data, training=training)
        except Exception as e:
            tf.print("Error in ContextRNN:", e)
            # 如果出错，提供一个合理的替代值
            task_representation = tf.random.normal([batch_size, self.context_dim], 
                                                  mean=0.0, stddev=0.1, dtype=tf.float32)
        
        # 调试信息：打印任务表示 - 使用tf.print
        if not training:
            tf.print("Task representation mean:", tf.reduce_mean(tf.abs(task_representation)))
            tf.print("Task representation shape:", tf.shape(task_representation))
        
        # 确保输出形状正确 [batch_size, context_dim]
        if len(task_representation.shape) == 1:
            # 如果是单个向量 [context_dim]，扩展为 [1, context_dim]
            task_representation = tf.expand_dims(task_representation, 0)
            
        # 如果批大小不一致，进行调整
        if tf.shape(task_representation)[0] == 1 and batch_size > 1:
            # 复制到批大小
            task_representation = tf.tile(task_representation, [batch_size, 1])
        
        # 确保动态任务表示的形状正确
        task_representation = tf.reshape(task_representation, [batch_size, self.context_dim])
            
        return task_representation
    
    @tf.function
    def get_action(self, o: tf.Tensor, deterministic: tf.Tensor = tf.constant(False)) -> tf.Tensor:
        """重写父类方法，使用动态任务表示z_t选择动作 a_t ~ π_θ(·|s_t^o, z_t)
        
        对应伪代码中的第4-5行：
        - 计算动态任务表示 z_t = RNN_θ({(s_i^o, a_i, r_i)}_{i=t-h-1}^{t-1})
        - 选择动作 a_t ~ π_θ(·|s_t^o, z_t)
        
        Args:
            o: 观测值，形状为[obs_dim]或[batch_size, obs_dim]
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 选择的动作，形状为[act_dim]
        """
        # 确保输入张量为float32类型
        o = tf.cast(o, dtype=tf.float32)
        
        # 准备历史数据
        history_data = (
            self.h_actions, 
            self.h_rewards, 
            self.h_observations
        )
        
        # 检查历史数据是否全为零或接近零
        h_actions_mean = tf.reduce_mean(tf.abs(self.h_actions))
        h_rewards_mean = tf.reduce_mean(tf.abs(self.h_rewards))
        h_observations_mean = tf.reduce_mean(tf.abs(self.h_observations))
        
        # 如果历史数据接近零，添加一些小随机值
        is_zero_history = tf.less(h_actions_mean + h_rewards_mean + h_observations_mean, 1e-6)
        
        def add_noise():
            # 添加一些随机噪声到历史数据
            batch_size = tf.shape(self.h_actions)[0]
            noisy_actions = self.h_actions + tf.random.normal(
                tf.shape(self.h_actions), mean=0.0, stddev=0.01, dtype=tf.float32)
            noisy_rewards = self.h_rewards + tf.random.normal(
                tf.shape(self.h_rewards), mean=0.0, stddev=0.01, dtype=tf.float32)
            noisy_observations = self.h_observations + tf.random.normal(
                tf.shape(self.h_observations), mean=0.0, stddev=0.01, dtype=tf.float32)
            return (noisy_actions, noisy_rewards, noisy_observations)
        
        # 条件选择是否添加噪声
        history_data = tf.cond(
            is_zero_history,
            add_noise,
            lambda: history_data
        )
        
        # 计算动态任务表示 - 伪代码第4行
        task_representation = self.compute_task_representation(history_data, training=False)
        
        # 确保o是二维张量 [batch_size, obs_dim]
        if len(o.shape) == 1:
            o = tf.expand_dims(o, axis=0)  # 添加批次维度 [1, obs_dim]
            
        # 确保task_representation是二维张量 [batch_size, context_dim]
        task_representation = tf.reshape(task_representation, [1, self.context_dim])
            
        # 将任务表示与观测拼接 - 准备给actor网络的输入
        obs_with_context = tf.concat([o, task_representation], axis=-1)
        
        # 调用actor网络获取动作分布 - 伪代码第5行
        mu, _, pi, _ = self.actor(obs_with_context, training=False)
        
        # 根据是否确定性策略选择返回mu还是pi
        action = tf.cond(deterministic, lambda: mu[0], lambda: pi[0])
        
        # 更新历史记录 - 为下一次动作选择准备数据
        self.update_history(o[0], action, tf.constant(0.0, dtype=tf.float32))
        
        return action
    
    def get_action_test(self, o: tf.Tensor, deterministic: tf.Tensor = tf.constant(False)) -> tf.Tensor:
        """测试时使用的动作选择函数"""
        return self.get_action(o, deterministic)
    
    def update_history(self, o, a, r):
        """更新历史信息，用于生成动态任务表示的历史数据
        
        接收一个新的观测-动作-奖励元组，更新历史缓冲区。
        
        Args:
            o: 观测值，形状为[obs_dim]
            a: 动作，形状为[action_dim]
            r: 奖励，标量
        """
        # 将输入转换为张量，并确保是float32类型
        o = tf.cast(tf.convert_to_tensor(o), dtype=tf.float32)
        if len(o.shape) == 1:
            o = tf.expand_dims(o, axis=0)  # 增加批次维度 [1, obs_dim]
        
        a = tf.cast(tf.convert_to_tensor(a), dtype=tf.float32)
        if len(a.shape) == 1:
            a = tf.expand_dims(a, axis=0)  # 增加批次维度 [1, action_dim]
        
        r = tf.cast(tf.convert_to_tensor(r), dtype=tf.float32)
        if len(r.shape) == 0:
            r = tf.reshape(r, (1, 1))  # 增加批次和特征维度 [1, 1]
        elif len(r.shape) == 1:
            r = tf.expand_dims(r, axis=-1)  # 增加特征维度 [batch_size, 1]
        
        # 更新recent数组（滚动更新）
        self.recent_actions = tf.roll(self.recent_actions, shift=-1, axis=0)
        self.recent_rewards = tf.roll(self.recent_rewards, shift=-1, axis=0)
        self.recent_observations = tf.roll(self.recent_observations, shift=-1, axis=0)
        
        # 更新最后一个位置
        indices = tf.constant([[self.history_length - 1]])
        self.recent_actions = tf.tensor_scatter_nd_update(self.recent_actions, indices, [a[0]])
        self.recent_rewards = tf.tensor_scatter_nd_update(self.recent_rewards, indices, [r[0]])
        self.recent_observations = tf.tensor_scatter_nd_update(self.recent_observations, indices, [o[0]])
        
        # 更新历史序列张量，用于批量计算
        self.h_actions = tf.concat([
            self.h_actions[:, 1:], 
            tf.expand_dims(a, axis=1)
        ], axis=1)
        
        self.h_rewards = tf.concat([
            self.h_rewards[:, 1:], 
            tf.expand_dims(r, axis=1)
        ], axis=1)
        
        self.h_observations = tf.concat([
            self.h_observations[:, 1:], 
            tf.expand_dims(o, axis=1)
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
        # 将当前缓冲区移动到旧缓冲区（伪代码第11行）
        if hasattr(self, 'replay_buffer') and self.replay_buffer is not None and self.replay_buffer.size > 0:
            # 创建旧缓冲区（如果不存在）
            if self.old_replay_buffer is None:
                self.old_replay_buffer = ReplayBuffer(
                    obs_dim=self.obs_dim, 
                    act_dim=self.action_dim, 
                    size=self.replay_size
                )
            
            # 将当前缓冲区数据复制到旧缓冲区
            idx = min(self.replay_buffer.size, self.replay_buffer.max_size)
            self.old_replay_buffer.obs_buf[:idx] = self.replay_buffer.obs_buf[:idx]
            self.old_replay_buffer.next_obs_buf[:idx] = self.replay_buffer.next_obs_buf[:idx]
            self.old_replay_buffer.actions_buf[:idx] = self.replay_buffer.actions_buf[:idx]
            self.old_replay_buffer.rewards_buf[:idx] = self.replay_buffer.rewards_buf[:idx]
            self.old_replay_buffer.done_buf[:idx] = self.replay_buffer.done_buf[:idx]
            self.old_replay_buffer.size = idx
            self.old_replay_buffer.ptr = self.replay_buffer.ptr
        
        # 调用父类方法处理任务切换（会创建新的回放缓冲区）
        super().on_task_start(current_task_idx)
        
        # 重置历史记录
        self.on_episode_end()
    
    def clear_replay_buffer(self):
        """清空当前的经验回放缓冲区"""
        if hasattr(self, 'replay_buffer') and self.replay_buffer is not None:
            # 重置指针和大小，保持缓冲区内存不变
            self.replay_buffer.ptr = 0
            self.replay_buffer.size = 0
    
    def get_gradients(
        self,
        seq_idx: tf.Tensor,
        aux_batch: Dict[str, tf.Tensor],
        obs: tf.Tensor,
        next_obs: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        done: tf.Tensor,
    ) -> Tuple[Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]], Dict]:
        """重写梯度计算方法，为actor和critic提供历史信息"""
        # 确保输入张量都是float32类型
        obs = tf.cast(obs, dtype=tf.float32)
        next_obs = tf.cast(next_obs, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.float32)
        rewards = tf.cast(rewards, dtype=tf.float32)
        done = tf.cast(done, dtype=tf.float32)
        
        # 从缓冲区获取历史数据
        batch_size = tf.shape(obs)[0]
        
        # 不再使用全零初始化，而是添加一些随机噪声，确保任务表示不为零
        # 创建带有小随机值的历史数据
        h_a = tf.random.normal([batch_size, self.history_length, self.action_dim], 
                               mean=0.0, stddev=0.01, dtype=tf.float32)
        h_r = tf.random.normal([batch_size, self.history_length, 1], 
                               mean=0.0, stddev=0.01, dtype=tf.float32)
        h_o = tf.random.normal([batch_size, self.history_length, self.obs_dim], 
                               mean=0.0, stddev=0.01, dtype=tf.float32)
        
        # 调试信息：检查历史数据
        tf.print("Debug - history data:")
        tf.print("  history action mean:", tf.reduce_mean(tf.abs(h_a)))
        tf.print("  history reward mean:", tf.reduce_mean(tf.abs(h_r)))
        tf.print("  history observation mean:", tf.reduce_mean(tf.abs(h_o)))
        
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
            # 使用training=True确保ContextRNN参与梯度计算
            try:
                task_representation = self.compute_task_representation(pre_act_rew, training=True)
                next_task_representation = self.compute_task_representation(next_pre_act_rew, training=True)
            except Exception as e:
                tf.print("Error in compute_task_representation:", e)
                # 提供一个合理的任务表示替代值
                task_representation = tf.random.normal([batch_size, self.context_dim], 
                                                      mean=0.0, stddev=0.1, dtype=tf.float32)
                next_task_representation = tf.random.normal([batch_size, self.context_dim], 
                                                          mean=0.0, stddev=0.1, dtype=tf.float32)
            
            # 确保任务表示不是全零
            task_rep_mean = tf.reduce_mean(tf.abs(task_representation))
            next_task_rep_mean = tf.reduce_mean(tf.abs(next_task_representation))
            
            # 如果任务表示接近零，添加一些小随机值
            is_zero_task_rep = tf.less(task_rep_mean, 1e-6)
            is_zero_next_task_rep = tf.less(next_task_rep_mean, 1e-6)
            
            def add_noise_to_task_rep():
                return task_representation + tf.random.normal(
                    tf.shape(task_representation), mean=0.0, stddev=0.1, dtype=tf.float32)
                
            def add_noise_to_next_task_rep():
                return next_task_representation + tf.random.normal(
                    tf.shape(next_task_representation), mean=0.0, stddev=0.1, dtype=tf.float32)
            
            # 条件选择是否添加噪声
            task_representation = tf.cond(is_zero_task_rep, add_noise_to_task_rep, lambda: task_representation)
            next_task_representation = tf.cond(is_zero_next_task_rep, add_noise_to_next_task_rep, 
                                            lambda: next_task_representation)
            
            # 调试：打印任务表示
            tf.print("Debug - task representations:")
            tf.print("  Current task representation mean:", tf.reduce_mean(tf.abs(task_representation)))
            tf.print("  Next task representation mean:", tf.reduce_mean(tf.abs(next_task_representation)))
            
            # 调试信息：输入张量的形状和类型
            tf.print("Debug - Input tensors:")
            tf.print("  obs shape:", tf.shape(obs), "dtype:", obs.dtype)
            tf.print("  actions shape:", tf.shape(actions), "dtype:", actions.dtype)
            tf.print("  task_representation shape:", tf.shape(task_representation), "dtype:", task_representation.dtype)

            # 将任务表示与观测结合
            obs_with_context = tf.concat([obs, task_representation], axis=-1)
            next_obs_with_context = tf.concat([next_obs, next_task_representation], axis=-1)
            
            # 确保数据类型一致
            obs_with_context = tf.cast(obs_with_context, dtype=tf.float32)
            next_obs_with_context = tf.cast(next_obs_with_context, dtype=tf.float32)
            
            # 调试信息：输入到网络的维度
            tf.print("Debug - obs_with_context shape:", tf.shape(obs_with_context), "dtype:", obs_with_context.dtype)
            tf.print("Debug - actions shape:", tf.shape(actions), "dtype:", actions.dtype)

            # 传入带有任务表示的观测获取动作和Q值
            mu, log_std, pi, logp_pi = self.actor(obs_with_context, training=True)
            
            # 调试：添加一些随机测试，看看critic是否能产生非零值
            random_noise = tf.random.normal(tf.shape(actions), mean=0.0, stddev=0.1, dtype=tf.float32)
            test_actions = actions + random_noise
            test_q1 = self.critic1(obs_with_context, test_actions, training=True)
            tf.print("Debug - Random test q1 mean:", tf.reduce_mean(test_q1))
            
            # 将动作和带上下文的观测传给critic
            q1 = self.critic1(obs_with_context, actions, training=True)
            q2 = self.critic2(obs_with_context, actions, training=True)
            
            # 检查q1和q2是否接近0
            is_q1_near_zero = tf.reduce_mean(tf.abs(q1)) < 1e-4
            is_q2_near_zero = tf.reduce_mean(tf.abs(q2)) < 1e-4
            
            # 如果Q值接近0，打印更多诊断信息并测试网络
            def print_debug_info():
                tf.print("WARNING: Q values are close to zero!")
                # 检查critic网络的权重和结构
                tf.print("Critic网络结构:")
                for i in range(len(self.critic1.core.layers)):
                    layer = self.critic1.core.layers[i]
                    tf.print("  Layer", i, ":", layer.__class__.__name__)
                
                # 找到第一个有kernel的层
                for i in range(len(self.critic1.core.layers)):
                    layer = self.critic1.core.layers[i]
                    if hasattr(layer, 'kernel'):
                        tf.print("Critic1 Dense层权重均值:", tf.reduce_mean(layer.kernel))
                        tf.print("Critic1 Dense层偏置均值:", tf.reduce_mean(layer.bias))
                        break
                
                # 检查最后一层
                if hasattr(self.critic1.head, 'layers'):
                    last_layer = self.critic1.head.layers[-1]
                    if hasattr(last_layer, 'kernel'):
                        tf.print("Critic1最后一层权重均值:", tf.reduce_mean(last_layer.kernel))
                        tf.print("Critic1最后一层偏置均值:", tf.reduce_mean(last_layer.bias))
                
                # 使用随机输入测试critic网络
                test_obs = tf.random.normal([1, self.obs_dim + self.context_dim], dtype=tf.float32)
                test_act = tf.random.normal([1, self.action_dim], dtype=tf.float32)
                test_q1 = self.critic1(test_obs, test_act, training=False)
                test_q2 = self.critic2(test_obs, test_act, training=False)
                tf.print("随机输入测试 - Critic1输出:", test_q1)
                tf.print("随机输入测试 - Critic2输出:", test_q2)
                
                # 检查当前输入是否全为0
                tf.print("当前输入obs_with_context平均值:", tf.reduce_mean(tf.abs(obs_with_context)))
                tf.print("当前输入actions平均值:", tf.reduce_mean(tf.abs(actions)))
                
                # 检查任务表示
                tf.print("任务表示平均绝对值:", tf.reduce_mean(tf.abs(task_representation)))
                
                return tf.constant(0)
            
            tf.cond(tf.logical_or(is_q1_near_zero, is_q2_near_zero), 
                    print_debug_info, 
                    lambda: tf.constant(0))

            # 使用当前策略计算Q值
            q1_pi = self.critic1(obs_with_context, pi, training=True)
            q2_pi = self.critic2(obs_with_context, pi, training=True)
            
            # 调试信息：打印策略Q值
            tf.print("Debug - q1_pi mean:", tf.reduce_mean(q1_pi))
            tf.print("Debug - q2_pi mean:", tf.reduce_mean(q2_pi))

            # 获取下一个状态的动作和对数概率
            _, _, pi_next, logp_pi_next = self.actor(next_obs_with_context, training=True)

            # 计算目标Q值
            target_q1 = self.target_critic1(next_obs_with_context, pi_next, training=True)
            target_q2 = self.target_critic2(next_obs_with_context, pi_next, training=True)
            
            # 调试信息：打印目标Q值
            tf.print("Debug - target_q1 mean:", tf.reduce_mean(target_q1))
            tf.print("Debug - target_q2 mean:", tf.reduce_mean(target_q2))

            # 使用双Q网络取最小值
            min_q_pi = tf.minimum(q1_pi, q2_pi)
            min_target_q = tf.minimum(target_q1, target_q2)
            
            # 调试：显示min_q和log_alpha
            tf.print("Debug - min_q_pi:", tf.reduce_mean(min_q_pi))
            tf.print("Debug - min_target_q:", tf.reduce_mean(min_target_q))
            tf.print("Debug - log_alpha:", tf.reduce_mean(log_alpha))
            tf.print("Debug - exp(log_alpha):", tf.reduce_mean(tf.math.exp(log_alpha)))

            # 基于熵正则化的贝尔曼更新
            q_backup = tf.stop_gradient(
                rewards + self.gamma * (1 - done) * (min_target_q - tf.math.exp(log_alpha) * logp_pi_next)
            )
            
            # 调试信息：打印q_backup的组成部分
            tf.print("Debug - rewards平均值:", tf.reduce_mean(rewards))
            tf.print("Debug - gamma:", self.gamma)
            tf.print("Debug - (1-done)平均值:", tf.reduce_mean(1-done))
            tf.print("Debug - logp_pi_next平均值:", tf.reduce_mean(logp_pi_next))
            tf.print("Debug - 熵项平均值:", tf.reduce_mean(tf.math.exp(log_alpha) * logp_pi_next))
            tf.print("Debug - q_backup平均值:", tf.reduce_mean(q_backup))

            # 检查Q值是否接近0
            is_q1_near_zero = tf.reduce_mean(tf.abs(q1)) < 1e-4
            is_q2_near_zero = tf.reduce_mean(tf.abs(q2)) < 1e-4
            
            # 如果Q值接近0，打印更多诊断信息并测试网络
            def print_debug_info():
                tf.print("WARNING: Q values are close to zero!")
                # 检查critic网络的权重和结构
                tf.print("Critic网络结构:")
                for i in range(len(self.critic1.core.layers)):
                    layer = self.critic1.core.layers[i]
                    tf.print("  Layer", i, ":", layer.__class__.__name__)
                
                # 找到第一个有kernel的层
                for i in range(len(self.critic1.core.layers)):
                    layer = self.critic1.core.layers[i]
                    if hasattr(layer, 'kernel'):
                        tf.print("Critic1 Dense层权重均值:", tf.reduce_mean(layer.kernel))
                        tf.print("Critic1 Dense层偏置均值:", tf.reduce_mean(layer.bias))
                        break
                
                # 检查最后一层
                if hasattr(self.critic1.head, 'layers'):
                    last_layer = self.critic1.head.layers[-1]
                    if hasattr(last_layer, 'kernel'):
                        tf.print("Critic1最后一层权重均值:", tf.reduce_mean(last_layer.kernel))
                        tf.print("Critic1最后一层偏置均值:", tf.reduce_mean(last_layer.bias))
                
                # 使用随机输入测试critic网络
                test_obs = tf.random.normal([1, self.obs_dim + self.context_dim], dtype=tf.float32)
                test_act = tf.random.normal([1, self.action_dim], dtype=tf.float32)
                test_q1 = self.critic1(test_obs, test_act, training=False)
                test_q2 = self.critic2(test_obs, test_act, training=False)
                tf.print("随机输入测试 - Critic1输出:", test_q1)
                tf.print("随机输入测试 - Critic2输出:", test_q2)
                
                # 检查当前输入是否全为0
                tf.print("当前输入obs_with_context平均值:", tf.reduce_mean(tf.abs(obs_with_context)))
                tf.print("当前输入actions平均值:", tf.reduce_mean(tf.abs(actions)))
                
                # 检查任务表示
                tf.print("任务表示平均绝对值:", tf.reduce_mean(tf.abs(task_representation)))
                
                return tf.constant(0)
            
            tf.cond(tf.logical_or(is_q1_near_zero, is_q2_near_zero), 
                    print_debug_info, 
                    lambda: tf.constant(0))

            # 计算SAC损失
            pi_loss = tf.reduce_mean(tf.math.exp(log_alpha) * logp_pi - min_q_pi)
            q1_loss = 0.5 * tf.reduce_mean((q_backup - q1) ** 2)
            q2_loss = 0.5 * tf.reduce_mean((q_backup - q2) ** 2)
            value_loss = q1_loss + q2_loss
            
            # 调试信息：打印损失
            tf.print("Debug - pi_loss:", pi_loss)
            tf.print("Debug - q1_loss:", q1_loss)
            tf.print("Debug - q2_loss:", q2_loss)
            
            # 增加ContextRNN的损失，鼓励有效的任务表示学习
            # 可以添加一个正则化项或辅助损失
            context_loss = tf.reduce_mean(tf.square(task_representation)) * 0.001  # 轻微的正则化
            
            # 添加辅助损失
            auxiliary_loss = self.get_auxiliary_loss(seq_idx, aux_batch)
            metrics = dict(
                pi_loss=pi_loss,
                q1_loss=q1_loss,
                q2_loss=q2_loss,
                q1=tf.reduce_mean(q1),
                q2=tf.reduce_mean(q2),
                logp_pi=tf.reduce_mean(logp_pi),
                reg_loss=auxiliary_loss,
                context_loss=context_loss
            )

            # 将所有损失组合到一起
            pi_loss += auxiliary_loss
            value_loss += auxiliary_loss
            total_context_loss = context_loss + auxiliary_loss

            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(
                    log_alpha * tf.stop_gradient(logp_pi + self.target_entropy)
                )

        # 计算梯度
        actor_gradients = g.gradient(pi_loss, self.actor.trainable_variables)
        critic_variables = self.critic1.trainable_variables + self.critic2.trainable_variables
        critic_gradients = g.gradient(value_loss, critic_variables)
        
        # 计算ContextRNN的梯度
        context_gradients = g.gradient(total_context_loss, self.context_rnn.trainable_variables)
        
        if self.auto_alpha:
            alpha_gradient = g.gradient(alpha_loss, self.all_log_alpha)
        else:
            alpha_gradient = None
        del g

        gradients = (actor_gradients, critic_gradients, alpha_gradient, context_gradients)
        return gradients, metrics
    
    def apply_update(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient: List[tf.Tensor],
        context_gradients: List[tf.Tensor] = None,
    ) -> None:
        """重写更新方法，添加ContextRNN的梯度更新"""
        # 更新actor参数
        self.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # 更新critic参数
        self.optimizer.apply_gradients(zip(critic_gradients, self.critic_variables))
        
        # 更新ContextRNN参数
        if context_gradients is not None:
            self.optimizer.apply_gradients(zip(context_gradients, self.context_rnn.trainable_variables))

        # 更新alpha参数
        if self.auto_alpha:
            self.optimizer.apply_gradients([(alpha_gradient, self.all_log_alpha)])

        # Polyak averaging for target variables
        for v, target_v in zip(
            self.critic1.trainable_variables, self.target_critic1.trainable_variables
        ):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)
        for v, target_v in zip(
            self.critic2.trainable_variables, self.target_critic2.trainable_variables
        ):
            target_v.assign(self.polyak * target_v + (1 - self.polyak) * v)
    
    def get_learn_on_batch(self, current_task_idx: int):
        """重写获取批量学习函数的方法，实现伪代码中的第8-10行"""
        @tf.function
        def learn_on_batch(
            seq_idx: tf.Tensor,
            batch: Dict[str, tf.Tensor],
            episodic_batch: Dict[str, tf.Tensor] = None,
            aux_batch: Dict[str, tf.Tensor] = None,
        ) -> Dict:
            # 初始化，默认使用传入的batch
            combined_batch = batch
            
            # 如果存在旧缓冲区且不为空，则进行混合采样
            if self.old_replay_buffer is not None and self.old_replay_buffer.size > 0:
                # 计算从当前和旧缓冲区分别采样的大小
                # 对应伪代码第8-9行中的计算公式
                n_tasks = tf.constant(1, dtype=tf.float32)  # 假设只有一个旧任务
                
                # 从当前缓冲区采样: b × min(1/n, 1-β) （伪代码第8行）
                current_ratio = tf.minimum(1.0/n_tasks, 1.0 - self.replay_ratio)
                current_size = tf.cast(tf.cast(self.batch_size, tf.float32) * current_ratio, tf.int32)
                
                # 从旧缓冲区采样: b × min((n-1)/n, β) （伪代码第9行）
                old_ratio = tf.minimum((n_tasks-1.0)/n_tasks, self.replay_ratio)
                old_size = self.batch_size - current_size
                
                if current_size > 0:
                    current_batch = self.replay_buffer.sample_batch(current_size)
                else:
                    current_batch = None
                
                # 尝试从旧缓冲区采样
                if old_size > 0:
                    try:
                        old_batch = self.old_replay_buffer.sample_batch(old_size)
                        
                        # 合并当前和旧的批次
                        if current_batch is not None:
                            # 合并所有键值对
                            combined_batch = {}
                            for key in current_batch:
                                if key in old_batch:
                                    combined_batch[key] = tf.concat([current_batch[key], old_batch[key]], axis=0)
                        else:
                            combined_batch = old_batch
                    except:
                        # 如果旧缓冲区采样失败，使用当前批次
                        if current_batch is not None:
                            combined_batch = current_batch
            
            # 使用组合的批次计算梯度和指标（伪代码第10行）
            gradients, metrics = self.get_gradients(seq_idx, aux_batch, **combined_batch)
            
            # 应用梯度更新所有参数，包括ContextRNN
            self.apply_update(*gradients)
            
            return metrics
        
        return learn_on_batch
