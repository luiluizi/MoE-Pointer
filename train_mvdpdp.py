import os
import sys
import socket
import setproctitle
import torch
import wandb
import yaml
import numpy as np

from pathlib import Path
from config import get_config
from envs.mvdpdp.env import DroneTransferEnv
from envs.mvdpdp.env_lade import DroneTransferEnvLADE
from runner.mvdpdp_runner import MVDPDPRunner as Runner
from utils.util import get_logger


def make_env(all_args, env_args, device, is_train):
    assert all_args.env_name == "mvdpdp"
    batch_size = all_args.n_rollout_threads if is_train else all_args.eval_episodes
    kwargs = {
        "env_args": env_args,
        "batch_size": batch_size,
        "device": device,
        "algorithm": all_args.algorithm,
        "is_train": is_train,
        "use_ar": not all_args.not_use_ar,
        "seed": all_args.seed,
        "seed_env": all_args.seed_env,
    }
    if "lade" in env_args["scenario"]:
        env = DroneTransferEnvLADE(**kwargs, city=all_args.dataset)
    else:
        env = DroneTransferEnv(**kwargs)
    if not is_train:
        all_args.eval_episode = env.batch_size
    return env

def parse_args(args, parser):
    all_args = parser.parse_args()

    return all_args

def acquire_lock(lock_dir_path):
    try:
        os.makedirs(lock_dir_path)
        return True
    except FileExistsError:
        return False

def create_run_dir(run_dir):
    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')] + [-1]
    for i in range(max(exst_run_nums) + 1, 1919810):
        curr_run = 'run%i' % i
        if acquire_lock(str(run_dir / curr_run)):
            return curr_run
    raise RuntimeError("can't create run dir")

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    with open(all_args.env_config_path, "r", encoding="utf-8") as _file:
        env_args = yaml.load(_file, yaml.Loader)
    env_args["deliveryed_visible"] = all_args.algorithm in ["mapdp"]
    env_args["max_consider_requests"] = -1 if all_args.algorithm in ["mapdp", "prob_heuristic", "sa", "ga", "nearest", "cpast"] else env_args["max_consider_requests"]
    if env_args["scenario"] == "lade":
        env_args["scenario"] = f"lade-{all_args.dataset}"
    elif "2D" not in env_args["scenario"]:
        all_args.not_use_node_emb = False
    all_args.episode_length = env_args["n_frame"]
    if not all_args.only_eval:
        print(yaml.dump(all_args, indent=4))
        print(yaml.dump(env_args, indent=4))
        logger = get_logger(not all_args.only_eval)
        logger.info(yaml.dump(all_args, indent=4))
        logger.info(yaml.dump(env_args, indent=4))
    all_args.env_args = env_args
    assert all_args.algorithm in ["mapt", "mapdp"] or all_args.only_eval, "only mapt needs training."

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        # 这里改一下：如果有指定的gpu id则使用该gpu，没有的话就使用第一个gpu
        print("choose to use gpu...")
        device = torch.device(f"cuda:{all_args.gpu_id}")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path("/mnt/jfs6/g-bairui/results") / all_args.env_name / env_args["scenario"] / all_args.algorithm / str(all_args.gpu_id)
    if not all_args.only_eval and not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb and not all_args.only_eval:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm) + "_seed" + str(all_args.seed),
                         group=all_args.map_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    elif not all_args.only_eval:
        curr_run = create_run_dir(run_dir)
        run_dir = run_dir / curr_run
        print(f"output to {str(run_dir)}")
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        
        with (run_dir / "config.yaml").open("w") as file:
            yaml.dump(vars(all_args), file)

    setproctitle.setproctitle(str(all_args.algorithm) + "-" + str(all_args.env_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env
    envs = make_env(all_args, env_args, device, is_train=True) if not all_args.only_eval else None
    eval_envs = make_env(all_args, env_args, device, is_train=False) if not all_args.not_use_eval or all_args.only_eval else None

    config = {
        "all_args": all_args,
        "env_args": env_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir
    }

    runner = Runner(config)
    if not all_args.use_wandb and not all_args.only_eval:
        for k, v in vars(all_args).items():
            runner.writter.add_text(k, str(v))
        for k, v in env_args.items():
            runner.writter.add_text("env:" + k, str(v))

    if all_args.only_eval:
        runner.eval(None)
    else:
        runner.run()


if __name__ == "__main__":
    main(sys.argv[1:])