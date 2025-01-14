"""
Simple script that produces vectorized environments
`_make_<>()` creates a single instance of the env - private function
`make_<>_vector_env(n)` produces a vector of n envs to be run in parallel - public functions
"""

import gym

def _make_cartpole():
  """
  v1 ends the env after 500 reward is given
  """
  env = gym.make("CartPole-v1")
  env = gym.wrappers.RecordEpisodeStatistics(env)
  return env

def make_cartpole_vector_env(num_envs):
  return gym.vector.SyncVectorEnv([
      _make_cartpole
      for i in range(num_envs)])

def make_atari_vector_env(num_envs, envname):
  global atariname
  atariname = envname
  return gym.vector.SyncVectorEnv([
      _make_atari
      for i in range(num_envs)])

def _make_atari():
  """
  atariname is passed as a global variable to avoid currying
  standard Atari preprocessing w/ 4 frame stackings
  we used NoFrameskip v4 environments: 'CentipedeNoFrameskip-v4'
  """
  env = gym.wrappers.AtariPreprocessing(gym.make(atariname), scale_obs=True)
  env = gym.wrappers.RecordEpisodeStatistics(gym.wrappers.FrameStack(env, 4))
  return env