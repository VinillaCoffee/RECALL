import os
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from myrecall.sac.sac import SAC
from myrecall.sac import cotasp_model
from myrecall.dict_learning.task_dict import OnlineDictLearnerV2


class CoTASPSAC(SAC):
    def __init__(
        self,
        tasks: str,
        method: str,
        env: Any,
        test_envs: List[Any],
        logger: Any,
        actor_cl: type = cotasp_model.MlpActor,
        actor_kwargs: Dict = None,
        critic_cl: type = cotasp_model.MlpCritic,
        critic_kwargs: Dict = None,
        seed: int = 0,
        steps: int = 1_000_000,
        log_every: int = 20_000,
        replay_size: int = 1_000_000,
        gamma: float = 0.99,
        polyak: float = 0.995,
        lr: float = 1e-3,
        alpha: Union[float, str] = "auto",
        batch_size: int = 128,
        start_steps: int = 10_000,
        update_after: int = 1000,
        update_every: int = 50,
        num_test_eps_stochastic: int = 10,
        num_test_eps_deterministic: int = 1,
        max_episode_len: int = 200,
        reset_buffer_on_task_change: bool = True,
        buffer_type: Any = None,
        reset_optimizer_on_task_change: bool = False,
        reset_critic_on_task_change: bool = False,
        clipnorm: float = None,
        target_output_std: float = None,
        agent_policy_exploration: bool = False,
        # CoTASP specific parameters
        update_dict: bool = True,
        update_coef: bool = True,
        dict_configs: Dict = None,
        load_policy_dir: Optional[str] = None,
        load_dict_dir: Optional[str] = None,
    ):
        """CoTASP-SAC implementation based on the original SAC algorithm."""
        super(CoTASPSAC, self).__init__(
            tasks, method, env, test_envs, logger, actor_cl, actor_kwargs, critic_cl, 
            critic_kwargs, seed, steps, log_every, replay_size, gamma, polyak, lr, 
            alpha, batch_size, start_steps, update_after, update_every, 
            num_test_eps_stochastic, num_test_eps_deterministic, max_episode_len, 
            reset_buffer_on_task_change, buffer_type, reset_optimizer_on_task_change, 
            reset_critic_on_task_change, clipnorm, target_output_std, agent_policy_exploration
        )
        
        if dict_configs is None:
            dict_configs = {}
            
        self.update_dict = update_dict
        self.update_coef = update_coef
        
        # Initialize task encoder
        self.task_encoder = SentenceTransformer('all-MiniLM-L12-v2')
        self.task_embeddings = []
        
        # Initialize dictionary learners for each layer
        self.dict4layers = {}
        if hasattr(self.actor, 'hidden_dims'):
            hidden_dims = self.actor.hidden_dims
        else:
            # 如果actor没有明确的hidden_dims属性，从actor_kwargs中获取或使用默认值
            hidden_dims = actor_kwargs.get('hidden_dims', [256, 256])
            
        for id_layer, hidn in enumerate(hidden_dims):
            dict_learner = OnlineDictLearnerV2(
                384,  # 句子嵌入维度 (SentenceTransformer 'all-MiniLM-L12-v2' 的输出维度)
                hidn,
                seed + id_layer + 1,
                None,  # 是否使用svd字典初始化
                **dict_configs
            )
            self.dict4layers[f'embeds_bb_{id_layer}'] = dict_learner
            
        # 加载预训练的字典（如果提供）
        if load_dict_dir is not None:
            for k in self.dict4layers.keys():
                self.dict4layers[k].load(f'{load_dict_dir}/{k}.pkl')
                
        # 初始化累积掩码
        self.cumul_masks = {}
        # TODO: 在任务开始时初始化masks
        
        # 加载预训练的策略（如果提供）
        if load_policy_dir is not None and os.path.exists(load_policy_dir):
            self.actor.load_weights(load_policy_dir)
    
    def on_task_start(self, current_task_idx: int) -> None:
        """当新任务开始时进行处理"""
        super().on_task_start(current_task_idx)
        
        # Get task description - 这里假设env有一个get_task_description方法
        if hasattr(self.env, 'get_task_description'):
            description = self.env.get_task_description(current_task_idx)
        else:
            description = f"Task {current_task_idx}"
            
        # 编码任务嵌入
        task_embedding = self.task_encoder.encode(description)[np.newaxis, :]
        self.task_embeddings.append(task_embedding)
        
        # 为任务设置初始alpha（字典系数）
        for layer_name, dict_learner in self.dict4layers.items():
            alpha_l = dict_learner.get_alpha(task_embedding)
            
            # 更新模型的嵌入层 - 这里需要根据具体的模型架构调整
            # 这里假设actor模型有一个设置任务系数的方法
            if hasattr(self.actor, 'set_task_coefficients'):
                self.actor.set_task_coefficients(current_task_idx, layer_name, alpha_l)
    
    def on_task_end(self, current_task_idx: int) -> None:
        """当任务结束时更新字典"""
        super().on_task_end(current_task_idx)
        
        # 获取当前任务的masks
        if hasattr(self.actor, 'get_masks'):
            current_masks = self.actor.get_masks(current_task_idx)
            
            # 更新累积掩码
            if not self.cumul_masks:
                self.cumul_masks = current_masks
            else:
                for k in current_masks:
                    if k in self.cumul_masks:
                        self.cumul_masks[k] = tf.math.maximum(self.cumul_masks[k], current_masks[k])
                    else:
                        self.cumul_masks[k] = current_masks[k]
                        
            # 根据累积掩码更新梯度掩码
            if hasattr(self.actor, 'set_grad_masks'):
                self.actor.set_grad_masks(self.cumul_masks)
        
        # 更新字典学习器
        dict_stats = {}
        if self.update_dict:
            for layer_name in self.dict4layers.keys():
                if hasattr(self.actor, 'get_task_coefficients'):
                    # 获取优化后的alpha
                    optimal_alpha_l = self.actor.get_task_coefficients(current_task_idx, layer_name)
                    optimal_alpha_l = np.array([optimal_alpha_l.flatten()])
                    task_embedding = self.task_embeddings[current_task_idx]
                    
                    # 通过CD在线更新字典
                    self.dict4layers[layer_name].update_dict(optimal_alpha_l, task_embedding)
                    
                    dict_stats[layer_name] = {
                        'sim_mat': self.dict4layers[layer_name]._compute_overlapping(),
                        'change_of_d': np.array(self.dict4layers[layer_name].change_of_dict)
                    }
        else:
            for layer_name in self.dict4layers.keys():
                dict_stats[layer_name] = {
                    'sim_mat': self.dict4layers[layer_name]._compute_overlapping(),
                    'change_of_d': 0
                }
                
        # 记录字典学习的统计信息
        for layer_name, stats in dict_stats.items():
            self.logger.store({
                f"dict/{layer_name}/change": np.mean(stats['change_of_d']),
                f"dict/{layer_name}/overlap": np.mean(stats['sim_mat'])
            })
            
        # 重置critic和优化器（如有需要）
        if self.reset_critic_on_task_change:
            self._reset_critics()
            
    def _reset_critics(self):
        """重置critic网络"""
        # 重置critic1和critic2的权重
        if hasattr(self.critic1, 'reset_weights'):
            self.critic1.reset_weights()
            self.target_critic1.set_weights(self.critic1.get_weights())
            
        if hasattr(self.critic2, 'reset_weights'):
            self.critic2.reset_weights()
            self.target_critic2.set_weights(self.critic2.get_weights())
    
    def get_action(self, o: tf.Tensor, deterministic: tf.Tensor = tf.constant(False)) -> tf.Tensor:
        """获取动作，考虑当前活跃的任务"""
        # 获取当前任务索引
        current_task_idx = getattr(self.env, "cur_seq_idx", 0)
        
        # 如果actor有get_action_for_task方法，使用它
        if hasattr(self.actor, 'get_action_for_task'):
            return self.actor.get_action_for_task(o, current_task_idx, deterministic)
        else:
            # 否则使用基类的方法
            return super().get_action(o, deterministic)
    
    def get_action_test(self, o: tf.Tensor, deterministic: tf.Tensor = tf.constant(False)) -> tf.Tensor:
        """用于测试的动作获取"""
        return self.get_action(o, deterministic)
    
    def adjust_gradients(
        self,
        actor_gradients: List[tf.Tensor],
        critic_gradients: List[tf.Tensor],
        alpha_gradient: List[tf.Tensor],
        current_task_idx: int,
        metrics: dict,
        episodic_batch: Dict[str, tf.Tensor] = None,
    ) -> Tuple[List[tf.Tensor], List[tf.Tensor], List[tf.Tensor]]:
        """根据CoTASP算法调整梯度"""
        # 如果不更新系数（alpha），则可能需要调整梯度
        if not self.update_coef and hasattr(self.actor, 'adjust_gradients_for_task'):
            actor_gradients = self.actor.adjust_gradients_for_task(
                actor_gradients, current_task_idx, self.cumul_masks)
            
        return actor_gradients, critic_gradients, alpha_gradient
    
    def save_model(self, save_dict_dir: Optional[str] = None):
        """保存模型和字典"""
        super().save_model()
        
        # 保存字典
        if save_dict_dir is not None:
            os.makedirs(save_dict_dir, exist_ok=True)
            for layer_name, dict_learner in self.dict4layers.items():
                dict_learner.save(f'{save_dict_dir}/{layer_name}.pkl') 