import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)

from typing import Callable, Iterable

from myrecall.envs import get_cl_env, get_single_env
from myrecall.sac.utils.logx import EpochLogger
from myrecall.sac import cotasp_model
from myrecall.tasks import TASK_SEQS
from myrecall.utils.enums import BufferType
from myrecall.utils.run_utils import get_sac_class
from myrecall.utils.utils import get_activation_from_str
from input_args import cl_parse_args

def print_env_info(env, name="环境"):
    """打印环境的详细信息"""
    print(f"\n=== {name} 信息 ===")
    print(f"环境类型: {env}")
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 尝试获取奖励范围
    try:
        print(f"奖励范围: {env.reward}")
    except AttributeError:
        print("奖励范围: 未定义")
    
    # 打印wrapper信息
    if hasattr(env, 'env'):
        print("\nWrapper信息:")
        current_env = env
        wrapper_chain = []
        while hasattr(current_env, 'env'):
            wrapper_chain.append(current_env.__class__.__name__)
            current_env = current_env.env
        wrapper_chain.append(current_env.__class__.__name__)
        
        for wrapper in wrapper_chain:
            print(f"- {wrapper}")
        
        # 打印OneHotAdder的信息
        if hasattr(env, 'env') and hasattr(env.env, 'env') and hasattr(env.env.env, 'env'):
            one_hot_adder = env.env.env.env
            if hasattr(one_hot_adder, 'one_hot_len'):
                print(f"\nOneHot编码信息:")
                print(f"编码长度: {one_hot_adder.one_hot_len}")
                print(f"原始观察空间: {one_hot_adder.orig_obs_space}")
                print(f"扩展后观察空间: {one_hot_adder.observation_space}")
    
    print("================\n")

if __name__ == "__main__":
    args = cl_parse_args()
    tasks = args.tasks
    steps_per_task = args.steps_per_task

    tasks_list = TASK_SEQS[tasks]
    print(f"\n任务列表: {tasks_list}")
    
    # 创建训练环境
    train_env = get_cl_env(tasks_list, steps_per_task)
    print_env_info(train_env, "训练环境")
    
    # 创建测试环境
    num_tasks = len(tasks_list)
    test_envs = [
        get_single_env(task, one_hot_idx=i, one_hot_len=num_tasks) for i, task in enumerate(tasks_list)
    ]
    print("\n=== 训练环境信息 ===")
    print_env_info(train_env, "训练环境")


    print("\n=== 测试环境信息 ===")
    for i, test_env in enumerate(test_envs):
        print_env_info(test_env, f"测试环境 {i} - {tasks_list[i]}")
    
    steps = steps_per_task * len(tasks_list)
    print(f"\n总训练步数: {steps}")
    print(f"每个任务的步数: {steps_per_task}")
    print(f"任务数量: {len(tasks_list)}")

