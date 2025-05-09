#!/usr/bin/env bash

# baselines
CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 --cl_method=ft

CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=ewc --cl_reg_coef=10000.0

CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=packnet --packnet_retrain_steps=100000 --clipnorm=2e-05

CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=pm --batch_size=512 --buffer_type=reservoir --reset_buffer_on_task_change=False --replay_size=6e6

CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=clonex --policy_reg_coef=100.0 --agent_policy_exploration=True --clipnorm=0.1

# proposed method
CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=recall --batch_size=128 --replay_size=6e6 \
  --behavior_cloning=True --policy_reg_coef=10.0 \
  --use_multi_layer_head=True --use_popArt=True --agent_policy_exploration=True --carried_critic=True


# pairwise tasks: pm
CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW0_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=pm --batch_size=512 --buffer_type=reservoir --reset_buffer_on_task_change=False --replay_size=2e6

# pairwise tasks: recall
CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW0_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=recall --batch_size=128 --replay_size=2e6 \
  --behavior_cloning=True --policy_reg_coef=0.01 \
  --use_popArt=True

# ablations: pm, pm+popart, pm+bc, recall
CUDA_VISIBLE_DEVICES=0 python3 run_cl.py --tasks=CW6_0 --seed=0 --steps_per_task=1e6 --log_every=2e4 \
  --cl_method=recall --batch_size=128 --replay_size=6e6 \
  --behavior_cloning=False --policy_reg_coef=10.0 \
  --use_multi_layer_head=True --use_popArt=False --agent_policy_exploration=True --carried_critic=True

# RECALL
python run_cl.py --tasks=CW10 --seed=0 --steps_per_task=5e5 --log_every=1e4 --cl_method=recall --batch_size=128 --replay_size=5e6 --behavior_cloning=True --policy_reg_coef=10.0 --use_multi_layer_head=True --use_popArt=True --agent_policy_exploration=True --carried_critic=True
                                                                                                                                          
# test:ClonEx-SAC
python run_cl.py --tasks=CW10 --seed=0 --steps_per_task=5e5 --log_every=1e4 --cl_method=clonex --policy_reg_coef=100.0 --agent_policy_exploration=True --clipnorm=0.1

# test:EWC
python run_cl.py --seed 0 --steps_per_task 5e5 --log_every 1e4 --tasks CW10 --cl_method ewc --cl_reg_coef 1e4 --logger_output tsv tensorboard

# test:PackNet
python run_cl.py --seed 0 --steps_per_task 5e5 --log_every 1e4 --tasks CW10 --cl_method packnet --packnet_retrain_steps 1e5 --clipnorm 1e-4 --logger_output tsv tensorboard

# test:FT
python run_cl.py --seed 0 --steps_per_task 5e5 --log_every 1e4 --tasks CW10 --cl_method ft --logger_output tsv tensorboard

# test:VCL
python run_cl.py --tasks=CW10 --seed=0 --steps_per_task=5e5 --log_every=1e4 --seed=0 --cl_method=vcl --cl_reg_coef=1.0 --vcl_first_task_kl=False --logger_output tsv tensorboard

#test:MTR
python run_cl.py --tasks=CW10 --seed=0 --steps_per_task=5e5 --log_every=1e4 --seed=0 --cl_method=mtr --reset_buffer_on_task_change=False --replay_size=5e6 --logger_output tsv tensorboard
#replay_size=1e6
python run_cl.py --tasks=CW10 --seed=0 --steps_per_task=5e5 --log_every=1e4 --seed=0 --cl_method=mtr --logger_output tsv tensorboard --reset_buffer_on_task_change=False

#test:ft
python run_cl.py --tasks=CW10 --seed=0 --steps_per_task=5e5 --log_every=1e4 --cl_method=ft

#test:3RL
python run_cl.py --tasks=CW10 --seed=0 --steps_per_task=5e5 --log_every=1e4 --cl_method=3rl --logger_output tsv tensorboard

#test:reservoir
python run_cl.py --tasks=CW10 --seed=0 --steps_per_task=5e5 --log_every=1e4 --cl_method=pm --buffer_type=reservoir --reset_buffer_on_task_change=False --replay_size=5e6
