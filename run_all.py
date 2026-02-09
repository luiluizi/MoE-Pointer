import argparse
import subprocess
import re
import itertools
import os
import sys
from multiprocessing import Pool
from multiprocess.pool import Pool
import yaml
import numpy as np
import time

import torch
import pandas as pd

pd.set_option('display.max_columns', 114514)  # 不限制列数显示
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', 114514)     # 不限制行数显示
pd.set_option('display.width', 114514)        # 自动确定显示宽度

datasets = pd.DataFrame(data=[
    ["synthetic-uniform", "envs/mvdpdp/env_uniform.yaml", None],
    # ["synthetic-2D", "envs/mvdpdp/env_2D.yaml", None],
    ["synthetic-cost", "envs/mvdpdp/env_cost.yaml", None],
    ["synthetic-large", "envs/mvdpdp/env_large.yaml", None],
    ["synthetic-large-cost", "envs/mvdpdp/env_large_cost.yaml", None],
    ["dhrd-tw", "envs/mvdpdp/env_dhrd.yaml", "tw"],
    ["dhrd-sg", "envs/mvdpdp/env_dhrd.yaml", "sg"],
    ["dhrd-se", "envs/mvdpdp/env_dhrd.yaml", "se"],
    ["synthetic-large2", "envs/mvdpdp/env_large2.yaml", None]
], columns=["Scenario", "env_config_path", "dataset"])

train_config = pd.DataFrame(data=[
    [1<<14, 256, 256, 1e-4, 128],
    # [1<<14, 256, 256, 1e-4, 128], # 效果甚微，已经调大 num_episodes
    [1<<14, 256, 256, 1e-4, 128],
    [1<<12, 64, 64, 2e-5, 128], # 效果甚微，已经调大 num_episodes
    [1<<12, 64, 64, 2e-5, 128], # 效果甚微，已经调大 num_episodes
    [1<<11, 64, 64, 2e-5, 32], # 不是很稳定，已经调小学习率。 # 发现效果比之前差一点点，又把学习率调回来了。
    [1<<11, 64, 64, 1e-5, 32], # 效果不怎么好，已经调大 num_episodes # 训练还不是很稳定 # 我不是很清楚为什么越 train 效果越差，已经调小 num_episodes，也调小了学习旅。
    [1<<13, 256, 256, 2e-5, 32],
    [1<<12, 64, 64, 2e-5, 128], # 未调
], columns=["num_episodes", "n_rollout_threads", "mini_batch_size", "lr", "eval_episodes"])

assert len(datasets) == len(train_config)
# train_config["num_episodes"] = 2
# train_config["n_rollout_threads"] = 2
# train_config["mini_batch_size"] = 2

train_algorithms = pd.DataFrame(data=[
    ["mapt", ],
    # ["mapdp", ],
], columns=["algorithm"])

# train ablation
all_ablation_config = pd.DataFrame(data=[
    [False, False, False, False, "mapt"],
    
    [True, False, False, False, "mapt-Rel"],
    [False, True, False, False, "mapt-AR"],
    # [True, True, False, False, "mapt-Rel-AR"],

    [True, False, True, True, "mapt-Rel-Heur"],
    # [False, True, True, True, "mapt-AR-Heur"],
    # [True, True, True, True, "mapt-Rel-AR-Heur"],
    
    # [False, False, True, False, "mapt-HeurReq"],
    # [False, False, False, True, "mapt-HeurVeh"],
    [False, False, True, True, "mapt-Heur"],
], columns=["not_use_relation", "not_use_ar", "not_use_heur_req", "not_use_heur_veh", "algorithm_label"])
all_ablation_config["algorithm"] = "mapt"
default_config = all_ablation_config.iloc[0].to_dict()
ablation_config = all_ablation_config.drop(index=0).reset_index(0, drop=True)

# train sensitive
all_sense_config = pd.DataFrame(data=[
    [0.00, 0.03, 0.5, 0.5, "mapt"],
], columns=["other_node_prob", "no_assign_prob", "load_balance_weight", "concentrat_weight", "algorithm_label"])
for col_idx, senses in enumerate([
    [0.02, 0.05], # optimized for dhrd-se
    [0.5, 0.7, 0.9], # optimized for env_uniform
    [0.0, 2.0, 4.0], # optimized for dhrd-se
    [0.0, 0.2, 0.4], # optimized for dhrd-se
]):
    col_name = all_sense_config.columns[col_idx]
    for s_value in senses:
        all_sense_config.loc[all_sense_config.__len__()] = all_sense_config.loc[0]
        all_sense_config.iloc[-1, col_idx] = s_value
        all_sense_config.iloc[-1, -1] = f"mapt-{col_name}-{s_value}"
all_sense_config["algorithm"] = "mapt"
default_sense_config = all_sense_config.iloc[0].to_dict()
sense_config = all_sense_config.drop(index=0).reset_index(0, drop=True)

all_sense_config_pb = all_sense_config.copy()
all_sense_config_pb["algorithm"] = all_sense_config["algorithm"].str.replace("mapt", "prob_heuristic")
all_sense_config_pb["algorithm_label"] = all_sense_config["algorithm_label"].str.replace("mapt", "prob_heuristic")
all_sense_config = pd.concat((all_sense_config, all_sense_config_pb))

eval_algorithms = pd.DataFrame(data=[
    ["cpsat", False],
    ["sa", False],
    ["ga", False],
    ["cpsat", True],
    ["sa", True],
    ["ga", True],
    ["nearest", None],
    ["prob_heuristic", None],
    ["mapdp", None],
    ["mapt", None],
], columns=["algorithm", "hindsight"])

def can_convert_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def retrieve_model(scenario, algorithm, config=default_config):
    config = {k: v for k, v in config.items() if k in ["not_use_relation", "not_use_ar", "not_use_heur_req", "not_use_heur_veh", "other_node_prob", "no_assign_prob", "load_balance_weight", "concentrat_weight"]}
    dir = f"results/mvdpdp/{scenario}/{algorithm}/"
    if not os.path.exists(dir):
        return None
    
    runids = [int(subdir[3:]) for subdir in os.listdir(dir) if subdir[:3] == "run"]
    for runid in sorted(runids, reverse=True):
        subdir = os.path.join(dir, f"run{runid}")
        with open(os.path.join(subdir, "config.yaml"), "r") as _f:
            loading_config = yaml.load(_f, Loader=yaml.SafeLoader)
        if any(loading_config.get(key, None) != value for key, value in config.items()):
            continue
        models_dir = os.path.join(subdir, "models")
        for model_name in sorted(os.listdir(models_dir), reverse=True):
            if "best" in model_name:
                return os.path.join(models_dir, model_name)
        models_id = [int(subdir[12:][:-3]) for subdir in os.listdir(models_dir) if subdir[:12] == "transformer_" and can_convert_to_int(subdir[12:][:-3])]
        # assert models_id.__len__() > 0
        if len(models_id) == 0:
            assert False, "Why there is a folder has no model?"
            return None
        else:
            model_id = max(models_id)
            model_path = os.path.join(models_dir, f"transformer_{model_id}.pt")
            return model_path
    
    return None

def series2cmd(s: pd.Series):
    args = []
    for name, value in s.items():
        if name in ["Scenario", "algorithm_label"]:
            continue
        if isinstance(value, bool):
            if value:
                args += [f"--{name}"]
        elif value is not None and (not isinstance(value, float) or not np.isnan(value)):
            args += [f"--{name}", str(value)]
    return args# + ["--seed", "3"]

def get_float(text, name):
    pattern = fr"{name}(-?\d*\.\d+|\d+)"
    return re.findall(pattern, text)

def get_string(text, name):
    pattern = fr"{name}([^\s]+)"
    return re.findall(pattern, text)

def wrapper(func):
    def _func(args):
        return func(*args)
    return _func

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--transfer", action="store_true", default=False)
    parser.add_argument("--cmd", action="store_true", default=False, help="only print train commands.")
    parser.add_argument("--ablation", action="store_true", default=False)
    parser.add_argument("--no_time", action="store_true", default=False)
    parser.add_argument("--sense", action="store_true", default=False)
    args = parser.parse_args()

    if args.train:
        n_gpus = torch.cuda.device_count()
        
        @wrapper
        def run(cmd, device_map):
            envs = os.environ.copy()
            if "CUDA_VISIBLE_DEVICES" in envs:
                device = envs["CUDA_VISIBLE_DEVICES"].split(",")[device_map[os.getpid()]]
            else:
                device = str(device_map[os.getpid()])
            envs["CUDA_VISIBLE_DEVICES"] = device
            time.sleep(list(device_map.keys()).index(os.getpid()))
            proc = subprocess.Popen([sys.executable, "train_mvdpdp.py"] + cmd, stdout=subprocess.PIPE, env=envs)
            output, error = proc.communicate()
            if proc.returncode != 0:
                return [None], [None], None
            # assert proc.returncode == 0

            output = output.decode("utf-8")
            reward = list(map(float, get_float(output, "eval_average_episode_rewards: ")))
            ratio = list(map(float, get_float(output, "eval_episode_serve_ratio_mean: ")))
            dir = get_string(output, "output to ")[0]

            return (reward, ratio, dir)
            

        with Pool(n_gpus) as pool:
            device_map = {process.pid: idx for idx, process in enumerate(pool._pool)}

            datasets["merge"] = 1
            train_algorithms["merge"] = 1
            all_config = train_algorithms.merge(pd.concat((datasets, train_config), axis=1), on="merge")
            if args.ablation:
                all_config = all_config.merge(ablation_config, on="algorithm", how="left")
            if args.sense:
                all_config = all_config.merge(sense_config, on="algorithm", how="left")
            del datasets["merge"]
            del train_algorithms["merge"]
            del all_config["merge"]

            popens_args = [(series2cmd(config), device_map) for idx, config in all_config.iterrows()]
            if args.cmd:
                for popen_args in popens_args:
                    print(" ".join(popen_args[0]))
            else:
                results = list(pool.map(run, popens_args))
        
        if not args.cmd:
            for idx, (reward, ratio, dir) in zip(all_config.index, results):
                all_config.loc[idx, ["reward", "max_reward", "ratio", "max_ratio", "dir"]] = [reward[-1], max(reward), ratio[-1], max(ratio), dir]
        
        print(all_config)
        if not args.cmd:
            all_config.to_csv("train_results.txt", index=False, sep="\t")

    if args.eval:
        if args.ablation:
            eval_algorithms = eval_algorithms.merge(right=all_ablation_config, on="algorithm", how="left")
            eval_algorithms["algorithm_label"] = eval_algorithms["algorithm_label"].fillna(eval_algorithms["algorithm"], inplace=False)
        elif args.sense:
            eval_algorithms = eval_algorithms.merge(right=all_sense_config, on="algorithm", how="left")
            eval_algorithms["algorithm_label"] = eval_algorithms["algorithm_label"].fillna(eval_algorithms["algorithm"], inplace=False)
        else:
            eval_algorithms["algorithm_label"] = eval_algorithms["algorithm"]
        for idx in eval_algorithms.index:
            if eval_algorithms.loc[idx, "hindsight"]:
                eval_algorithms.loc[idx, "algorithm_label"] = eval_algorithms.loc[idx, "algorithm_label"] + "-Fore"
        
        res = pd.DataFrame(index=eval_algorithms["algorithm_label"], columns=pd.MultiIndex.from_tuples(itertools.product(datasets["Scenario"], ["Obj", "Comp", "Time"])))
        for _, dataset in datasets.iterrows():
            for _, _algorithm in eval_algorithms.iterrows():
                eval_config = pd.concat((dataset, _algorithm))
                algorithm = _algorithm["algorithm_label"]
                scenario = dataset["Scenario"]
                if algorithm == "cpsat" and not (scenario in ["synthetic-uniform", "synthetic-2D", "synthetic-cost"] or dataset["dataset"] == "se"):
                    res.loc[algorithm, (scenario, "Obj")] = None
                    res.loc[algorithm, (scenario, "Comp")] = None
                    res.loc[algorithm, (scenario, "Time")] = None
                    continue
                cur_ablation = default_config.copy()
                if args.ablation:
                    cur_ablation.update(_algorithm[["not_use_relation", "not_use_ar", "not_use_heur_req", "not_use_heur_veh"]].dropna().to_dict())
                if args.sense:
                    cur_ablation.update(_algorithm[["other_node_prob", "no_assign_prob", "load_balance_weight", "concentrat_weight"]].dropna().to_dict())
                eval_config["model_path"] = retrieve_model(scenario, _algorithm["algorithm"], cur_ablation)
                cmd = " ".join([sys.executable, "train_mvdpdp.py", "--only_eval"] + series2cmd(eval_config))
                print(cmd)
                if args.cmd:
                    continue
                # assert(False)
                continue
                proc = subprocess.Popen([sys.executable, "train_mvdpdp.py", "--only_eval"] + series2cmd(eval_config), stdout=subprocess.PIPE)
                output, error = proc.communicate()
                output = output.decode("utf-8")
                # assert proc.returncode == 0, [sys.executable, "train_mvdpdp.py", "--only_eval"] + series2cmd(eval_config)
                if proc.returncode != 0:
                    res.loc[algorithm, (scenario, "Obj")] = None
                    res.loc[algorithm, (scenario, "Comp")] = None
                    res.loc[algorithm, (scenario, "Time")] = None
                    print([sys.executable, "train_mvdpdp.py", "--only_eval"] + series2cmd(eval_config))
                    continue
                eval_average_episode_rewards = float(get_float(output, "eval_average_episode_rewards: ")[-1])
                eval_episode_serve_ratio_mean = float(get_float(output, "eval_episode_serve_ratio_mean: ")[-1])

                if not args.no_time:
                    proc = subprocess.Popen([sys.executable, "train_mvdpdp.py", "--only_eval", "--eval_episodes", "1"] + series2cmd(eval_config), stdout=subprocess.PIPE)
                    output, error = proc.communicate()
                    output = output.decode("utf-8")
                    assert proc.returncode == 0
                    eval_time = float(get_float(output, "eval_time: ")[-1])
                else:
                    eval_time = None

                print(eval_average_episode_rewards)
                res.loc[algorithm, (scenario, "Obj")] = eval_average_episode_rewards
                res.loc[algorithm, (scenario, "Comp")] = eval_episode_serve_ratio_mean
                res.loc[algorithm, (scenario, "Time")] = eval_time
                # res.loc[algorithm, (scenario, "model_path")] = eval_config["model_path"]

        print(res)
        if not args.cmd:
            res.to_csv("eval_results.txt", sep="\t")

    if args.transfer:
        res = pd.DataFrame(index=datasets["Scenario"], columns=pd.MultiIndex.from_tuples(itertools.product(datasets["Scenario"], ["Obj", "Comp"])))
        for _, dataset1 in datasets.iterrows():
            for _, dataset2 in datasets.iterrows():
                eval_config = dataset2.copy()
                eval_config["model_path"] = retrieve_model(dataset1["Scenario"], "mapt")

                cmd = " ".join([sys.executable, "train_mvdpdp.py", "--only_eval"] + series2cmd(eval_config))
                if args.cmd:
                    print(cmd)
                    continue

                proc = subprocess.Popen([sys.executable, "train_mvdpdp.py", "--only_eval"] + series2cmd(eval_config), stdout=subprocess.PIPE)
                output, error = proc.communicate()
                output = output.decode("utf-8")
                assert proc.returncode == 0
                eval_average_episode_rewards = float(get_float(output, "eval_average_episode_rewards: ")[-1])
                eval_episode_serve_ratio_mean = float(get_float(output, "eval_episode_serve_ratio_mean: ")[-1])

                res.loc[dataset1["Scenario"], (dataset2["Scenario"], "Obj")] = eval_average_episode_rewards
                res.loc[dataset1["Scenario"], (dataset2["Scenario"], "Comp")] = eval_episode_serve_ratio_mean
        
        print(res)
        if not args.cmd:
            res.to_csv("transfer_results.txt", sep="\t")