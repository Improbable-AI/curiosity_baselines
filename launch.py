import os
import sys
import subprocess
import time
import json
import argparse
from six.moves import shlex_quote
import torch

# Runners
from rlpyt.runners.minibatch_rl import MinibatchRl, MinibatchRlEval

# Policies
from rlpyt.agents.pg.atari import AtariFfAgent, AtariLstmAgent
from rlpyt.agents.pg.mujoco import MujocoFfAgent, MujocoLstmAgent

# Samplers
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector, CpuWaitResetCollector, CpuEvalCollector
from rlpyt.samplers.parallel.gpu.collectors import GpuResetCollector, GpuWaitResetCollector, GpuEvalCollector
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler

# Environments
from rlpyt.samplers.collections import TrajInfo
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.envs.mazeworld.mazeworld.envs.pycolab_env import PycolabTrajInfo
from rlpyt.envs.gym import make as gym_make
from rlpyt.envs.gym import mario_make, deepmind_make

# Learning Algorithms
from rlpyt.algos.pg.ppo import PPO
from rlpyt.algos.pg.a2c import A2C

# Utils
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.affinity import make_affinity, encode_affinity, affinity_from_code
from rlpyt.utils.launching.arguments import get_args
from rlpyt.utils.misc import wrap_print


with open('./global.json') as global_params:
    params = json.load(global_params)
    _WORK_DIR = params['local_workdir']
    _RESULTS_DIR = params['local_resultsdir']
    _TB_PORT = params['tb_port']
    _ATARI_ENVS = params['envs']['atari_envs']
    _MUJOCO_ENVS = params['envs']['mujoco_envs']


def launch_tmux(args):

    # determine log directory and argument string
    if args.pretrain is not None:
        log_dir = os.path.join(_RESULTS_DIR, args.pretrain)
        cmd_file = open(log_dir + '/cmd.txt')
        args_string = cmd_file.read().split(' ')
        args_string[args_string.index('-pretrain') + 1] = args.pretrain
        args_string = args_string[2:] # take out python3 launch.py
        args_string = ' '.join(args_string)
    else:
        name = '_'.join([args.alg, args.env])
        if os.path.isdir(f'{_RESULTS_DIR}/{name}/run_0'):
            runs = os.listdir(f'{_RESULTS_DIR}/{name}')
            try:
                runs.remove('tmp')
            except ValueError:
                pass
            try:
                runs.remove('.DS_Store')
            except ValueError:
                pass
            sorted_runs = sorted(runs, key=lambda run: int(run.split('_')[-1]))
            run_id = int(sorted_runs[-1].split('_')[-1]) + 1
        else:
            run_id = 0
            os.makedirs(os.path.join(_RESULTS_DIR, name, f'run_{run_id}'))
        log_dir = os.path.join(_RESULTS_DIR, name, f'run_{run_id}')

        args_string = ''
        for arg, value in vars(args).items():
            if arg == 'launch_tmux':
                args_string += '-launch_tmux no '
            elif value is None and arg == 'log_dir':
                args_string += f'-log_dir {log_dir} '
            elif value is True:
                args_string += f'-{arg} '
            elif value is False:
                pass
            else:
                args_string += f'-{arg} {value} '

    # check whether to run
    print('\n')
    print('#'*50)
    print('Generated command:')
    print('-'*50)
    print(f'python3 launch.py {args_string}')
    print('#'*50)
    print('\n')

    commands = {'htop' : 'htop',
                'runner' : f'python3 launch.py {args_string}',
                'tb' : f'tensorboard --logdir {log_dir} --port {_TB_PORT} --bind_all'}
    os.system(f'kill -9 $( lsof -i:{_TB_PORT} -t ) > /dev/null 2>&1')
    os.system('tmux kill-session -t experiment')
    os.system('tmux new-session -s experiment -n htop -d bash')
    i = 0
    for name, cmd in commands.items():
        if name != 'htop':
            os.system(f'tmux new-window -t experiment:{i+1} -n {name} bash')
        os.system(f'tmux send-keys -t experiment:{name} {shlex_quote(cmd)} Enter')
        i += 1

    # save arguments, and command if needed
    if args.pretrain is None:
        time.sleep(6) # wait for logdir to be created
        with open(log_dir + '/cmd.txt', 'w') as cmd_file:
            cmd_file.writelines(commands['runner'])
        with open(log_dir + '/git.txt', 'w') as git_file:
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
            git_file.write('{}/{}'.format(branch, commit))


def start_experiment(args):

    args_json = json.dumps(vars(args), indent=4)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)
    with open(args.log_dir + '/arguments.json', 'w') as jsonfile:
        jsonfile.write(args_json)

    config = dict(env_id=args.env)
    
    if args.sample_mode == 'gpu':
        assert args.num_gpus > 0
        affinity = dict(cuda_idx=0, workers_cpus=list(range(args.num_cpus)))
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    else:
        affinity = dict(workers_cpus=list(range(args.num_cpus)))

    # potentially reload models
    initial_optim_state_dict = None
    initial_model_state_dict = None
    if args.pretrain != 'None':
        os.system(f"find {args.log_dir} -name '*.json' -delete") # clean up json files for video recorder
        checkpoint = torch.load(os.path.join(_RESULTS_DIR, args.pretrain, 'params.pkl'))
        initial_optim_state_dict = checkpoint['optimizer_state_dict']
        initial_model_state_dict = checkpoint['agent_state_dict']

    # ----------------------------------------------------- POLICY ----------------------------------------------------- #
    model_args = dict(curiosity_kwargs=dict(curiosity_alg=args.curiosity_alg))

    model_args['curiosity_kwargs']['std_rew_scaling'] = args.std_rew_scaling

    if args.curiosity_alg =='icm':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['forward_loss_wt'] = args.forward_loss_wt
    elif args.curiosity_alg == 'disagreement':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['ensemble_size'] = args.ensemble_size
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['forward_loss_wt'] = args.forward_loss_wt
        model_args['curiosity_kwargs']['device'] = args.sample_mode
    elif args.curiosity_alg == 'ndigo':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['pred_horizon'] = args.pred_horizon
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['num_predictors'] = args.num_predictors
        model_args['curiosity_kwargs']['device'] = args.sample_mode
    elif args.curiosity_alg == 'rnd':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['prediction_beta'] = args.prediction_beta
        model_args['curiosity_kwargs']['drop_probability'] = args.drop_probability
        model_args['curiosity_kwargs']['gamma'] = args.discount
        model_args['curiosity_kwargs']['device'] = args.sample_mode
    elif args.curiosity_alg == 'rand':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['device'] = args.sample_mode

    # TODO MARIUS: Read input arguments from launch for Kohonen
    elif args.curiosity_alg == 'kohonen':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['device'] = args.sample_mode

    # TODO MARIUS: Read input arguments from launch for ART
    elif args.curiosity_alg == 'art':
        model_args['curiosity_kwargs']['feature_encoding'] = args.feature_encoding
        model_args['curiosity_kwargs']['batch_norm'] = args.batch_norm
        model_args['curiosity_kwargs']['rho'] = args.rho
        model_args['curiosity_kwargs']['alpha'] = args.alpha
        model_args['curiosity_kwargs']['beta'] = args.beta
        model_args['curiosity_kwargs']['device'] = args.sample_mode

    if args.env in _MUJOCO_ENVS:
        if args.lstm:
            agent = MujocoLstmAgent(initial_model_state_dict=initial_model_state_dict)
        else:
            agent = MujocoFfAgent(initial_model_state_dict=initial_model_state_dict)
    else:
        if args.lstm:
            agent = AtariLstmAgent(
                        initial_model_state_dict=initial_model_state_dict,
                        model_kwargs=model_args,
                        no_extrinsic=args.no_extrinsic
                        )
        else:
            agent = AtariFfAgent(initial_model_state_dict=initial_model_state_dict)

    # ----------------------------------------------------- LEARNING ALG ----------------------------------------------------- #
    if args.alg == 'ppo':
        if args.kernel_mu == 0.:
            kernel_params = None
        else:
            kernel_params = (args.kernel_mu, args.kernel_sigma)
        algo = PPO(
                discount=args.discount,
                learning_rate=args.lr,
                value_loss_coeff=args.v_loss_coeff,
                entropy_loss_coeff=args.entropy_loss_coeff,
                OptimCls=torch.optim.Adam,
                optim_kwargs=None,
                clip_grad_norm=args.grad_norm_bound,
                initial_optim_state_dict=initial_optim_state_dict, # is None is not reloading a checkpoint
                gae_lambda=args.gae_lambda,
                minibatches=args.minibatches, # if recurrent: batch_B needs to be at least equal, if not recurrent: batch_B*batch_T needs to be at least equal to this
                epochs=args.epochs,
                ratio_clip=args.ratio_clip,
                linear_lr_schedule=args.linear_lr,
                normalize_advantage=args.normalize_advantage,
                normalize_reward=args.normalize_reward,
                kernel_params=kernel_params,
                curiosity_type=args.curiosity_alg
                )
    elif args.alg == 'a2c':
        algo = A2C(
                discount=args.discount,
                learning_rate=args.lr,
                value_loss_coeff=args.v_loss_coeff,
                entropy_loss_coeff=args.entropy_loss_coeff,
                OptimCls=torch.optim.Adam,
                optim_kwargs=None,
                clip_grad_norm=args.grad_norm_bound,
                initial_optim_state_dict=initial_optim_state_dict,
                gae_lambda=args.gae_lambda,
                normalize_advantage=args.normalize_advantage
                )

    # ----------------------------------------------------- SAMPLER ----------------------------------------------------- #

    # environment setup
    traj_info_cl = TrajInfo # environment specific - potentially overriden below
    if 'mario' in args.env.lower():
        env_cl = mario_make
        env_args = dict(
            game=args.env,  
            no_extrinsic=args.no_extrinsic,
            no_negative_reward=args.no_negative_reward,
            normalize_obs=args.normalize_obs,
            normalize_obs_steps=10000
            )
    elif 'deepmind' in args.env.lower(): # pycolab deepmind environments
        env_cl = deepmind_make
        traj_info_cl = PycolabTrajInfo
        env_args = dict(
            game=args.env,
            no_extrinsic=args.no_extrinsic,
            no_negative_reward=args.no_negative_reward,
            normalize_obs=args.normalize_obs,
            normalize_obs_steps=10000,
            log_heatmaps=args.log_heatmaps,
            logdir=args.log_dir,
            obs_type=args.obs_type,
            max_steps_per_episode=args.max_episode_steps
            )
    elif args.env in _MUJOCO_ENVS:
        env_cl = gym_make
        env_args = dict(
            id=args.env, 
            no_extrinsic=args.no_extrinsic,
            no_negative_reward=args.no_negative_reward,
            normalize_obs=False,
            normalize_obs_steps=10000
            )
    elif args.env in _ATARI_ENVS:
        env_cl = AtariEnv
        traj_info_cl = AtariTrajInfo
        env_args = dict(
            game=args.env, 
            no_extrinsic=args.no_extrinsic,
            no_negative_reward=args.no_negative_reward,
            normalize_obs=args.normalize_obs,
            normalize_obs_steps=10000,
            downsampling_scheme='classical',
            record_freq=args.record_freq,
            record_dir=args.log_dir,
            horizon=args.max_episode_steps,
            )

    if args.sample_mode == 'gpu':
        if args.lstm:
            collector_class = GpuWaitResetCollector
        else:
            collector_class = GpuResetCollector
        sampler = GpuSampler(
            EnvCls=env_cl,
            env_kwargs=env_args,
            eval_env_kwargs=env_args,
            batch_T=args.timestep_limit,
            batch_B=args.num_envs,
            max_decorrelation_steps=0,
            TrajInfoCls=traj_info_cl,
            eval_n_envs=args.eval_envs,
            eval_max_steps=args.eval_max_steps,
            eval_max_trajectories=args.eval_max_traj,
            record_freq=args.record_freq,
            log_dir=args.log_dir,
            CollectorCls=collector_class
        )
    else:
        if args.lstm:
            collector_class = CpuWaitResetCollector
        else:
            collector_class = CpuResetCollector
        sampler = CpuSampler(
            EnvCls=env_cl,
            env_kwargs=env_args,
            eval_env_kwargs=env_args,
            batch_T=args.timestep_limit, # timesteps in a trajectory episode
            batch_B=args.num_envs, # environments distributed across workers
            max_decorrelation_steps=0,
            TrajInfoCls=traj_info_cl,
            eval_n_envs=args.eval_envs,
            eval_max_steps=args.eval_max_steps,
            eval_max_trajectories=args.eval_max_traj,
            record_freq=args.record_freq,
            log_dir=args.log_dir,
            CollectorCls=collector_class
            )

    # ----------------------------------------------------- RUNNER ----------------------------------------------------- #
    if args.eval_envs > 0:
        runner = MinibatchRlEval(
            algo=algo,
            agent=agent,
            sampler=sampler,
            n_steps=args.iterations,
            affinity=affinity,
            log_interval_steps=args.log_interval,
            log_dir=args.log_dir,
            pretrain=args.pretrain
            )
    else:
        runner = MinibatchRl(
            algo=algo,
            agent=agent,
            sampler=sampler,
            n_steps=args.iterations,
            affinity=affinity,
            log_interval_steps=args.log_interval,
            log_dir=args.log_dir,
            pretrain=args.pretrain
            )

    with logger_context(args.log_dir, config, snapshot_mode="last", use_summary_writer=True):
        runner.train()


if __name__ == "__main__":

    args = get_args()
    if args.launch_tmux == 'yes':
        launch_tmux(args) # launches tmux
    else:
        start_experiment(args) # launches the actual experiment inside of a tmux session









