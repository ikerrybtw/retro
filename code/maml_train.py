from sandbox.rocky.tf.algos.maml_trpo import MAMLTRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from maml_examples.point_env_randgoal import PointEnvRandGoal
from maml_examples.point_env_randgoal_oracle import PointEnvRandGoalOracle
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.rocky.tf.policies.maml_minimal_gauss_mlp_policy import MAMLGaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.envs.gym_env import GymEnv

import tensorflow as tf

learning_rate = 1e-2

fast_learning_rate = 0.1
# baseline = 'linear'
fast_batch_size = 20  # 10 works for [0.1, 0.2], 20 doesn't improve much for [0,0.2]
meta_batch_size = 40  # 10 also works, but much less stable, 20 is fairly stable, 40 is more stable
max_path_length = 50 # 0
num_grad_updates = 1
meta_step_size = 0.01
train_env = 'FetchReach-v1'
use_maml = True

stub(globals())
env = TfEnv(normalize(GymEnv(train_env, record_video = False, force_reset = True, record_log = False)))
print(env)
policy = MAMLGaussianMLPPolicy(
                name="policy",
                env_spec=env.spec,
                grad_step_size=fast_learning_rate,
                hidden_nonlinearity=tf.nn.relu,
                hidden_sizes=(100,100,100),
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
                n_itr=400,
                use_maml=use_maml,
                step_size=meta_step_size,
                plot=False,
            )
run_experiment_lite(
                algo.train(),
                n_parallel=1,
                snapshot_mode="last",
                log_dir = "./data/",
                python_command='python3',
                seed=1,
                exp_prefix='vpg_maml_point100',
                exp_name='trpomaml'+str(int(use_maml))+'_fbs'+str(fast_batch_size)+'_mbs'+str(meta_batch_size)+'_flr_' + str(fast_learning_rate) + 'metalr_' + str(meta_step_size) +'_step1'+str(num_grad_updates),
                plot=False,
            )
