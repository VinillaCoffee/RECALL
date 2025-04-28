import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class MetaPolicyNetwork(tf.keras.Model):
    """Meta-Policy Network for CoTASP algorithm.
    
    This model generates task-specific policies based on task embeddings.
    """
    
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dims=[256, 256],
        activation='relu',
        output_activation=None,
        action_space=None,
        task_num=10,
        state_dependent_std=True,
        squash_output=True
    ):
        super(MetaPolicyNetwork, self).__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dims = hidden_dims
        self.task_num = task_num
        self.state_dependent_std = state_dependent_std
        self.squash_output = squash_output
        
        # Initialize embeddings for each layer and task
        self.embeddings = {}
        for i, dim in enumerate(hidden_dims):
            layer_name = f'embeds_bb_{i}'
            self.embeddings[layer_name] = tf.Variable(
                tf.zeros([task_num, dim]),
                trainable=True,
                name=f'{layer_name}_embedding'
            )
            
        # Policy network backbone layers
        self.hidden_layers = []
        for i, dim in enumerate(hidden_dims):
            self.hidden_layers.append(
                layers.Dense(
                    dim,
                    activation=activation,
                    name=f'hidden_{i}'
                )
            )
        
        # Output layers
        self.mu_layer = layers.Dense(
            act_dim,
            activation=output_activation,
            name='mu_layer'
        )
        
        if state_dependent_std:
            self.log_std_layer = layers.Dense(
                act_dim,
                activation=None,
                name='log_std_layer'
            )
        else:
            self.log_std = tf.Variable(
                tf.zeros([act_dim]),
                trainable=True,
                name='log_std'
            )
            
        # Set action scaling
        if action_space is not None:
            self.action_scale = action_space.high[0]
        else:
            self.action_scale = 1.0
            
        # Initialize masks for gradient masking
        self.masks = {}
        for i in range(len(hidden_dims)):
            self.masks[f'layer_{i}'] = tf.zeros([hidden_dims[i]])
            
    def call(self, obs, task_id, deterministic=False, with_logprob=True):
        """Forward pass of the network.
        
        Args:
            obs: Observation tensor
            task_id: Task identifier tensor
            deterministic: If True, return deterministic actions
            with_logprob: If True, compute log probabilities
            
        Returns:
            Tuple of (mu, log_std, pi, logp_pi) where:
                mu: Mean of the policy distribution
                log_std: Log standard deviation of the policy
                pi: Sampled action
                logp_pi: Log probability of the sampled action
        """
        batch_size = tf.shape(obs)[0]
        task_idx = task_id[0]  # Assuming task_id is a single-element tensor
        
        # Process through hidden layers with task-specific embeddings
        x = obs
        masks = {}
        for i, layer in enumerate(self.hidden_layers):
            layer_name = f'embeds_bb_{i}'
            alpha = tf.gather(self.embeddings[layer_name], task_idx)
            mask = tf.cast(alpha > 0, tf.float32)
            masks[f'layer_{i}'] = mask
            
            h = layer(x)
            # Apply task-specific modulation
            h = h * tf.expand_dims(mask, 0)
            x = h
            
        # Get mean and log_std
        mu = self.mu_layer(x)
        
        if self.state_dependent_std:
            log_std = self.log_std_layer(x)
            log_std = tf.clip_by_value(log_std, -20, 2)
        else:
            log_std = tf.broadcast_to(self.log_std, tf.shape(mu))
            
        std = tf.exp(log_std)
        
        # Sample from normal distribution
        pi_distribution = tf.random.normal(shape=tf.shape(mu), mean=mu, stddev=std)
        
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution
            
        if self.squash_output:
            # Apply tanh squashing
            pi_action = tf.tanh(pi_action)
            # Scale to action space
            pi_action = self.action_scale * pi_action
            
        # Compute log probabilities if required
        if with_logprob:
            if self.squash_output:
                # Compute log prob for squashed normal
                logp_pi = self._squashed_normal_logprob(pi_distribution, log_std, pi_action)
            else:
                # Standard normal log prob
                logp_pi = -0.5 * (
                    tf.reduce_sum(tf.square((pi_action - mu) / std), axis=-1)
                    + 2 * tf.reduce_sum(log_std, axis=-1)
                    + self.act_dim * tf.math.log(2 * np.pi)
                )
        else:
            logp_pi = None
            
        # Return the policy outputs and masks
        return mu, log_std, pi_action, logp_pi, {'masks': masks, 'means': mu, 'stddev': std}
        
    def _squashed_normal_logprob(self, x, log_std, squashed_x):
        """Compute log probability for squashed normal distribution."""
        # Standard normal log prob
        logp = -0.5 * (
            tf.reduce_sum(tf.square(x), axis=-1)
            + self.act_dim * tf.math.log(2 * np.pi)
            + 2 * tf.reduce_sum(log_std, axis=-1)
        )
        
        # Correction for tanh squashing
        correction = tf.reduce_sum(
            tf.math.log(1 - tf.square(tf.tanh(x)) + 1e-6), axis=-1
        )
        
        return logp - correction
    
    def get_action_for_task(self, obs, task_id, deterministic=False):
        """Get action for a specific task."""
        obs_tensor = tf.expand_dims(obs, 0)
        task_tensor = tf.constant([task_id])
        
        mu, log_std, pi, _, _ = self.call(obs_tensor, task_tensor, deterministic)
        
        return pi[0]
    
    def get_masks(self, task_id):
        """Get activation masks for a specific task."""
        masks = {}
        for i in range(len(self.hidden_dims)):
            layer_name = f'embeds_bb_{i}'
            alpha = tf.gather(self.embeddings[layer_name], task_id)
            mask = tf.cast(alpha > 0, tf.float32)
            masks[f'layer_{i}'] = mask
            
        return masks
    
    def set_task_coefficients(self, task_id, layer_name, alpha):
        """Set coefficients (alpha) for a specific task and layer."""
        if layer_name in self.embeddings:
            # Reshape alpha if needed to match the expected shape
            alpha_flat = tf.reshape(alpha, [-1])
            alpha_tensor = tf.convert_to_tensor(alpha_flat, dtype=tf.float32)
            
            # Update the embedding for the specific task
            update_op = tf.tensor_scatter_nd_update(
                self.embeddings[layer_name],
                tf.constant([[task_id]]),
                tf.expand_dims(alpha_tensor, 0)
            )
            self.embeddings[layer_name].assign(update_op)
            
    def get_task_coefficients(self, task_id, layer_name):
        """Get coefficients (alpha) for a specific task and layer."""
        if layer_name in self.embeddings:
            return tf.gather(self.embeddings[layer_name], task_id)
        return None
    
    def set_grad_masks(self, cumul_masks):
        """Set gradient masks based on cumulative masks."""
        self.grad_masks = cumul_masks
        
    def adjust_gradients_for_task(self, grads, task_id, cumul_masks):
        """Adjust gradients based on masks to only update unused parameters."""
        adjusted_grads = []
        
        # Process gradients for each variable
        for i, grad in enumerate(grads):
            var_name = self.trainable_variables[i].name
            
            # For embedding variables, allow full gradient update
            if any(name in var_name for name in self.embeddings.keys()):
                adjusted_grads.append(grad)
                continue
                
            # For hidden layers, mask gradients based on cumulative masks
            for j in range(len(self.hidden_dims)):
                if f'hidden_{j}' in var_name:
                    mask_key = f'layer_{j}'
                    if mask_key in cumul_masks:
                        # 创建反掩码 (1-mask)，使得只有未使用的神经元接收梯度
                        inverse_mask = 1.0 - cumul_masks[mask_key]
                        
                        # 根据是权重还是偏置应用掩码
                        if 'kernel' in var_name:  # 权重
                            # 对于权重，掩码需要扩展为适当的形状
                            mask_expanded = tf.expand_dims(inverse_mask, 0)
                            # 输出维度掩码
                            adjusted_grad = grad * tf.transpose(mask_expanded)
                        elif 'bias' in var_name:  # 偏置
                            # 对于偏置，直接应用掩码
                            adjusted_grad = grad * inverse_mask
                        else:
                            adjusted_grad = grad
                            
                        adjusted_grads.append(adjusted_grad)
                        break
            else:
                # For layers not needing masking, keep gradients unchanged
                adjusted_grads.append(grad)
                
        return adjusted_grads 