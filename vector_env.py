import gym

def _make_cartpole():
  #def curry():
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
  #return curry

def make_cartpole_vector_env(num_envs):
  return gym.vector.SyncVectorEnv([
      _make_cartpole
      for i in range(num_envs)])