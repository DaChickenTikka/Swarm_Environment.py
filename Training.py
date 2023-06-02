import tensorflow as tf
import Environment
import PIL.Image

#  from tf_agents.agents.dqn import dqn_agent
#  from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
#  from tf_agents.eval import metric_utils
#  from tf_agents.metrics import tf_metrics
#  from tf_agents.networks import sequential
#  from tf_agents.policies import py_tf_eager_policy
#  from tf_agents.policies import random_tf_policy
#  from tf_agents.replay_buffers import reverb_replay_buffer
#  from tf_agents.replay_buffers import reverb_utils
#  from tf_agents.trajectories import trajectory
#  from tf_agents.specs import tensor_spec
#  #from tf_agents.utils import common

if __name__ == '__main__':
    env = suite_gym.load(Environment)
    env.reset()
    PIL.Image.fromarray(env.render())
# num_iterations = 20000  # @param {type:"integer"}

# initial_collect_steps = 100  # @param {type:"integer"}
# collect_steps_per_iteration = 1  # @param {type:"integer"}
# replay_buffer_max_length = 100000  # @param {type:"integer"}

#  batch_size = 64  # @param {type:"integer"}
#  learning_rate = 1e-3  # @param {type:"number"}
#  log_interval = 200  # @param {type:"integer"}

#  num_eval_episodes = 10  # @param {type:"integer"}
#  eval_interval = 1000  # @param {type:"integer"}
