import argparse


def get_config():
    parser = argparse.ArgumentParser(description='madpdp', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm", type=str, default='moe_pointer', choices=["moe_pointer", "prob_heuristic", "cpsat", "nearest", "ga", "sa", "mapdp"])
    parser.add_argument("--hindsight", action="store_true", default=False, help="use hindsight policy.")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--seed_env", type=int, default=1, help="Random seed for env construction")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic", action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int, default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=64, help="Number of parallel envs for training rollouts")
    parser.add_argument("--num_episodes", type=int, default=8192, help='Number of environment steps to train (default: 10e6)')
    parser.add_argument("--user_name", type=str, default='xxx',help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_false', default=False, help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='mvdpdp', help="specify the name of environment")
    parser.add_argument("--env_config_path", type=str, default='./envs/mvdpdp/env_uniform.yaml', help="specify the path of environment config.")

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=1/4, help="warmup epoch ratio.")
    parser.add_argument("--opti_eps", type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--not_use_linear_lr_decay", action='store_true', default=False, help='not use a linear schedule on the learning rate')
    parser.add_argument("--use_cosine_lr_decay", action='store_true', default=False, help='use a cosine schedule on the learning rate')
    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=1, help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss", action='store_true', default=False, help="by default, don't clip loss value. If set, do clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.1, help='ppo clip parameter.')
    parser.add_argument("--mini_batch_size", type=int, default=4, help='ppo mini_batch_size')
    parser.add_argument("--entropy_coef", type=float, default=0.1, help='entropy term coefficient')
    parser.add_argument("--value_loss_coef", type=float, default=1, help='value loss coefficient')
    parser.add_argument("--use_max_grad_norm", action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help='max norm of gradients')
    parser.add_argument("--not_use_gae", action='store_true', default=False, help='don\'t use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.99, help='gae lambda parameter')
    parser.add_argument("--use_huber_loss", action='store_true', default=False, help="use huber loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # save and log parameters
    parser.add_argument("--save_interval", type=int, default=4, help="time duration between contiunous twice models saving.")
    parser.add_argument("--log_interval", type=int, default=1, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--only_eval", action="store_true", default=False, help="only evaluation then exit.")
    parser.add_argument("--not_use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int, default=1, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=64, help="number of episodes of a single evaluation.")
    parser.add_argument("--model_path", type=str, default=None, help="by default None. set the path to pretrained model.")

    # add for transformer
    parser.add_argument("--encode_state", action='store_true', default=False)
    parser.add_argument("--n_enc_block", type=int, default=5)
    parser.add_argument("--n_dec_block", type=int, default=3)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=2)
    
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--not_use_valuenorm", action='store_true', default=False, help="don't use running mean and std to normalize rewards.")
    
    parser.add_argument("--not_use_heur_req", action="store_true", default=False, help="don't use request heuristic prob.")
    parser.add_argument("--use_nearest_station", action="store_true", default=False, help="use nearest station assignment.")
    parser.add_argument("--not_use_heur_veh", action="store_true", default=False, help="don't use vehicle heuristic prob.")
    parser.add_argument("--not_use_ar", action="store_true", default=False, help="don't use auto regressive.")
    parser.add_argument("--not_use_relation", action="store_true", default=False, help="don't use relation-aware attention.")
    parser.add_argument("--not_use_node_emb", action="store_true", default=False, help="don't use node embedding for real dataset")
    parser.add_argument("--use_unbind_decode", action="store_true", default=False, help="don't bind emb and action as decoder input.")
    parser.add_argument("--use_moe", action="store_true", default=False, help="use MoE structure.")
    parser.add_argument("--check_grad", action="store_true", default=False, help="check grad is finite")

    # dataset
    parser.add_argument("--dataset", type=str, default="hz", choices=["hz", "sh", "cq"], help="real world dataset.")

    # hyper params
    parser.add_argument("--other_node_prob", type=float, default=0.00)
    parser.add_argument("--no_assign_prob", type=float, default=0.00)
    parser.add_argument("--load_balance_weight", type=float, default=0.5)
    return parser
