from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.envs.grid_world_env_rand import GridWorldEnvRand
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from sandbox.rocky.tf.policies.maml_minimal_categorical_mlp_policy import MAMLCategoricalMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
import tensorflow as tf

from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
import baselines.ppo2.ppo2 as ppo2
import baselines.ppo2.policies as policies
import gym_remote.exceptions as gre

from sonic_util import make_env

def main():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True # pylint: disable=E1101
  with tf.Session(config=config):
  # env = TfEnv(normalize(GridWorldEnvRand('four-state')))
    env = DummyVecEnv([make_env])
    policy = MAMLCategoricalMLPPolicy(
            name="policy",
            env_spec=env.spec,
            grad_step_size=fast_learning_rate,
            prob_network = nature_cnn
            hidden_nonlinearity=tf.nn.relu,
            hidden_sizes=(100,100),
        )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = MAMLTRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=fast_batch_size, # number of trajs for grad update
            max_path_length=max_path_length,
            meta_batch_size=meta_batch_size,
            num_grad_updates=num_grad_updates,
            n_itr=800,
            use_maml=use_maml,
            step_size=meta_step_size,
            plot=False,
        )
    run_experiment_lite(
            algo.train(),
            n_parallel=4,
            snapshot_mode="last",
            seed=1,
            exp_prefix='trpo_maml_4state',
            exp_name='trpo_maml'+str(int(use_maml))+'_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size)+'_flr_' + str(fast_learning_rate) + 'metalr_' + str(meta_step_size) +'_step1'+str(num_grad_updates),
            plot=False,
        )
if __name__ == '__main__':
    try:
        main()
    except gre.GymRemoteError as exc:
        print('exception', exc)
