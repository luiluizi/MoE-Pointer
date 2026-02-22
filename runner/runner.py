import time
import os

import wandb
import torch
from tensorboardX import SummaryWriter

from utils.replay_buffer import ReplayBuffer
from algorithms.mat_trainer import MATTrainer as TrainAlgo
from algorithms.pointer_transformer_policy import PointerTransformerPolicy
from algorithms.rolling_horizon_policy import RollingHorizonPolicy
from envs.env import DroneTransferEnv
from utils.util import get_logger

class DMPDPRunner:
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        self.all_args = config['all_args']
        self.logger = get_logger()
        self.env_args = config['env_args']
        self.envs: DroneTransferEnv = config['envs']
        self.eval_envs: DroneTransferEnv = config['eval_envs']
        self.device = config['device']     

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm = self.all_args.algorithm
        # Total number of training episodes
        self.num_episodes = self.all_args.num_episodes
        # Maximum number of steps per episode
        self.episode_length = self.all_args.episode_length
        # Number of parallel environments
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.use_linear_lr_decay = not self.all_args.not_use_linear_lr_decay and not self.all_args.use_cosine_lr_decay    
        self.use_cosine_lr_decay = self.all_args.use_cosine_lr_decay and self.all_args.not_use_linear_lr_decay
        
        self.warmup_epoch = self.all_args.warmup_ratio * self.num_episodes // self.n_rollout_threads
        self.logger.info(f"warmup_epoch {self.warmup_epoch}")
        self.use_wandb = self.all_args.use_wandb
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = not self.all_args.not_use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        
        # dir
        self.model_path = self.all_args.model_path
        self.logger.info("model path {}".format(self.model_path))
        self.logger.info("run dir " + str(config["run_dir"]))

        if self.use_wandb and not self.all_args.only_eval:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        elif not self.all_args.only_eval:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        # policy network
        if self.algorithm in ["moe_pointer", "mapdp"]:
            self.policy = PointerTransformerPolicy(self.all_args, self.env_args, device=self.device)
        elif self.algorithm == "prob_heuristic":
            self.policy = PointerTransformerPolicy(self.all_args, self.env_args, device=self.device, only_heuristic=True)
        elif self.algorithm in ["cpsat", "nearest", "ga", "sa"]:
            self.policy = RollingHorizonPolicy(self.all_args, use_hindsight=self.all_args.hindsight)
        else:
            raise AttributeError(f"unrecognized algorithm {self.algorithm}")

        if self.model_path is not None:
            self.restore(self.model_path)

        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, device=self.device)
        
        # buffer
        self.buffer = ReplayBuffer(self.all_args, self.envs, self.device)

    def run(self):
        self.warmup()
        start = time.time()
        n_episode_batch = self.num_episodes // self.n_rollout_threads

        train_episode_rewards = torch.zeros(self.n_rollout_threads, device=self.device)
        done_episodes_rewards = [] # heterogeneous

        one_episode_stage1_rewards = torch.zeros(self.n_rollout_threads, device=self.device)
        done_episode_stage1_rewards = []
        one_episode_stage2_rewards = torch.zeros(self.n_rollout_threads, device=self.device)
        done_episode_stage2_rewards = []
        one_episode_stage3_rewards = torch.zeros(self.n_rollout_threads, device=self.device)
        done_episode_stage3_rewards = []
        
        done_episode_stage1_ratio = []
        done_episode_stage2_ratio = []
        done_episode_stage3_ratio = []
        
        
        max_aver_reward = None

        for episode in range(n_episode_batch):
            start1 = time.time()
            if episode <= self.warmup_epoch:
                self.trainer.policy.lr_warmup(episode, self.warmup_epoch)
            elif self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, n_episode_batch, 'linear')
            elif self.use_cosine_lr_decay:
                self.trainer.policy.lr_decay(episode, n_episode_batch, 'cosine')
            for step in range(self.episode_length): 
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions)
                if dones.any():
                    assert dones.all()
                    obs = self.envs.reset()

                train_episode_rewards += rewards
                one_episode_stage1_rewards += infos["stage1_rewards"]
                one_episode_stage2_rewards += infos["stage2_rewards"]
                one_episode_stage3_rewards += infos["stage3_rewards"]
                
                if dones.any():
                    done_episodes_rewards.append(train_episode_rewards[dones].clone())
                    done_episode_stage1_rewards.append(one_episode_stage1_rewards[dones].clone())
                    done_episode_stage2_rewards.append(one_episode_stage2_rewards[dones].clone())
                    done_episode_stage3_rewards.append(one_episode_stage3_rewards[dones].clone())
                    done_episode_stage1_ratio.append(infos["global_finish_stage1_ratio"][dones].clone())
                    done_episode_stage2_ratio.append(infos["global_finish_stage2_ratio"][dones].clone())
                    done_episode_stage3_ratio.append(infos["global_finish_ratio"][dones].clone())

                one_episode_stage1_rewards[dones] = 0
                one_episode_stage2_rewards[dones] = 0
                one_episode_stage3_rewards[dones] = 0
                ###########
                train_episode_rewards[dones] = 0
                
                data = obs, rewards, dones, infos, \
                    values, actions, action_log_probs, \
                        rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)
                
            self.logger.info("collect down")
            # compute return and update network
            self.compute()
            train_infos = self.train()
            self.logger.info("train down")
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if ((episode + 1) % self.save_interval == 0 or episode == n_episode_batch - 1):
                self.save(episode)

            # log information
            if (episode + 1) % self.log_interval == 0:
                end = time.time()
                self.logger.info("\nScenario {} Algo {} updates {}/{} n_episode_batch, FPS {}."
                        .format(self.env_args["scenario"],
                                self.algorithm,
                                episode + 1,
                                n_episode_batch,
                                int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = torch.cat(done_episodes_rewards).mean().item()
                    self.logger.info(f"some episodes done, average rewards: {aver_episode_rewards}")
                    self.writter.add_scalars("train_episode_rewards", {"aver_rewards": aver_episode_rewards}, total_num_steps)
                    
                    #################
                    aver_episode_stage1_rewards = torch.cat(done_episode_stage1_rewards).mean().item()
                    aver_episode_stage2_rewards = torch.cat(done_episode_stage2_rewards).mean().item()
                    aver_episode_stage3_rewards = torch.cat(done_episode_stage3_rewards).mean().item()
                    aver_episode_stage1_ratio = torch.cat(done_episode_stage1_ratio).mean().item()
                    aver_episode_stage2_ratio = torch.cat(done_episode_stage2_ratio).mean().item()
                    aver_episode_stage3_ratio = torch.cat(done_episode_stage3_ratio).mean().item()
                    
                    self.logger.info(f"some episodes done, average stage1 rewards: {aver_episode_stage1_rewards}, stage2: {aver_episode_stage2_rewards}, stage3: {aver_episode_stage3_rewards}")
                    self.writter.add_scalars("train_episode_stage1_rewards", {"aver_rewards": aver_episode_stage1_rewards}, total_num_steps)
                    self.writter.add_scalars("train_episode_stage2_rewards", {"aver_rewards": aver_episode_stage2_rewards}, total_num_steps)
                    self.writter.add_scalars("train_episode_stage3_rewards", {"aver_rewards": aver_episode_stage3_rewards}, total_num_steps)
                    
                    self.logger.info(f"some episodes done, average stage1 ratio: {aver_episode_stage1_ratio}, stage2: {aver_episode_stage2_ratio}, stage3: {aver_episode_stage3_ratio}")
                    self.writter.add_scalars("train_episode_stage1_ratio", {"aver_ratio": aver_episode_stage1_ratio}, total_num_steps)
                    self.writter.add_scalars("train_episode_stage2_ratio", {"aver_ratio": aver_episode_stage2_ratio}, total_num_steps)
                    self.writter.add_scalars("train_episode_stage3_ratio", {"aver_ratio": aver_episode_stage3_ratio}, total_num_steps)
                    ###################
                    done_episodes_rewards = []
                    done_episode_stage1_rewards = []
                    done_episode_stage2_rewards = []
                    done_episode_stage3_rewards = []
                    
                    done_episode_stage1_ratio = []
                    done_episode_stage2_ratio = []
                    done_episode_stage3_ratio = []

            # eval
            if (episode + 1) % self.eval_interval == 0 and self.use_eval:
                eval_episode_rewards_mean = self.eval(total_num_steps)
                if max_aver_reward is None or eval_episode_rewards_mean > max_aver_reward:
                    max_aver_reward = eval_episode_rewards_mean
                    self.save(episode, is_best=True)
            end1 = time.time()
            self.logger.info(f"one epoch time {end1 - start1}" )

    def warmup(self):
        # reset env
        self.envs.seed(self.all_args.seed, self.all_args.seed_env)
        obs = self.envs.reset()
        self.buffer.obs[0] = obs

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_state, rnn_state_critic \
            = self.trainer.policy.get_actions(
                self.buffer.obs[step], self.buffer.rnn_states[step], self.buffer.rnn_states_critic[step], deterministic=False
            )

        return value, action, action_log_prob, rnn_state, rnn_state_critic

    def insert(self, data):
        obs, rewards, dones, infos, \
            values, actions, action_log_probs, \
                rnn_states, rnn_states_critic = data

        if rnn_states is not None:
            # use zeros tensor as initial rnn state.
            rnn_states[dones] = 0
            rnn_states_critic[dones] = 0

        self.buffer.insert(obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, ~dones)

    def log_train(self, train_infos, total_num_steps):
        # train_infos["average_step_rewards"] = self.buffer.rewards.mean().item()
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
    

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        
        eval_episode_stage1_rewards = []
        eval_episode_stage2_rewards = []
        eval_episode_stage3_rewards = []
        one_episode_stage1_rewards = torch.zeros(self.eval_envs.batch_size, device=self.device)
        one_episode_stage2_rewards = torch.zeros(self.eval_envs.batch_size, device=self.device)
        one_episode_stage3_rewards = torch.zeros(self.eval_envs.batch_size, device=self.device)
        
        eval_episode_courier_cost = []
        eval_episode_drone_cost = []
        eval_episode_profit = []
        
        one_episode_courier_cost = torch.zeros(self.eval_envs.batch_size, device=self.device)
        one_episode_drone_cost = torch.zeros(self.eval_envs.batch_size, device=self.device)
        one_episode_profit = torch.zeros(self.eval_envs.batch_size, device=self.device)
        
        eval_episode_serve_ratio = []
        eval_episode_serve_stage1_ratio = []
        eval_episode_serve_stage2_ratio = []
        one_episode_rewards = torch.zeros(self.eval_envs.batch_size, device=self.device)
        eval_time = 0

        self.eval_envs.seed(self.all_args.seed + self.all_args.only_eval + 1, self.all_args.seed_env)
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = torch.zeros(self.eval_envs.batch_size, self.recurrent_N, self.all_args.n_embd, device=self.device)

        if self.all_args.hindsight:
            t0 = time.time()
            self.policy.hindsight_update(**{
                "_global": self.eval_envs._global.copy(),
                "nodes": self.eval_envs.nodes.copy(),
                "couriers": self.eval_envs.couriers.copy(),
                "drones": self.eval_envs.drones.copy(),
                "requests": self.eval_envs.requests.copy(),
                "global_n_consider_requests": self.eval_envs.frame_n_accumulate_requests[:, -1]
            })
            eval_time += time.time() - t0

        while True:
            self.trainer.prep_rollout()
            t0 = time.time()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(eval_obs, eval_rnn_states, deterministic=True)
            eval_time += time.time() - t0

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions)
            eval_episode += eval_dones.sum()
            one_episode_rewards += eval_rewards
            one_episode_stage1_rewards += eval_infos["stage1_rewards"]
            one_episode_stage2_rewards += eval_infos["stage2_rewards"]
            one_episode_stage3_rewards += eval_infos["stage3_rewards"]
            one_episode_courier_cost += eval_infos["courier_cost"]
            one_episode_drone_cost += eval_infos["drone_cost"]
            one_episode_profit += eval_infos["profit"]
            if eval_dones.any():
                eval_episode_rewards.append(one_episode_rewards[eval_dones].clone())
                eval_episode_serve_ratio.append(eval_infos["global_finish_ratio"][eval_dones].clone())
                eval_episode_serve_stage1_ratio.append(eval_infos["global_finish_stage1_ratio"][eval_dones].clone())
                eval_episode_serve_stage2_ratio.append(eval_infos["global_finish_stage2_ratio"][eval_dones].clone())
                eval_episode_stage1_rewards.append(one_episode_stage1_rewards[eval_dones].clone())
                eval_episode_stage2_rewards.append(one_episode_stage2_rewards[eval_dones].clone())
                eval_episode_stage3_rewards.append(one_episode_stage3_rewards[eval_dones].clone())
                eval_episode_courier_cost.append(one_episode_courier_cost[eval_dones].clone())
                eval_episode_drone_cost.append(one_episode_drone_cost[eval_dones].clone())
                eval_episode_profit.append(one_episode_profit[eval_dones].clone())

            if eval_dones.any():
                assert eval_dones.all()

                eval_episode_rewards_mean = torch.cat(eval_episode_rewards).mean().item()
                eval_episode_rewards_max = torch.cat(eval_episode_rewards).max(0)[0].item()
                eval_episode_serve_ratio_mean = torch.cat(eval_episode_serve_ratio).mean().item()
                eval_episode_serve_stage1_ratio_mean = torch.cat(eval_episode_serve_stage1_ratio).mean().item()
                eval_episode_serve_stage2_ratio_mean = torch.cat(eval_episode_serve_stage2_ratio).mean().item()
                
                eval_episode_stage1_rewards_mean = torch.cat(eval_episode_stage1_rewards).mean().item()
                eval_episode_stage2_rewards_mean = torch.cat(eval_episode_stage2_rewards).mean().item()
                eval_episode_stage3_rewards_mean = torch.cat(eval_episode_stage3_rewards).mean().item()
                
                eval_episode_courier_cost_mean = torch.cat(eval_episode_courier_cost).mean().item()
                eval_episode_drone_cost_mean = torch.cat(eval_episode_drone_cost).mean().item()
                eval_episode_profit_mean = torch.cat(eval_episode_profit).mean().item()
                
                eval_env_infos = {
                    'eval_average_episode_rewards': eval_episode_rewards_mean,
                    'eval_max_episode_rewards': eval_episode_rewards_max,
                    "global_finish_ratio": eval_episode_serve_ratio_mean,
                    "global_finish_stage1_ratio": eval_episode_serve_stage1_ratio_mean,
                    "global_finish_stage2_ratio": eval_episode_serve_stage2_ratio_mean,
                }

                if not self.all_args.only_eval:
                    self.log_env(eval_env_infos, total_num_steps)
                # self.logger.info(f"eval_average_episode_rewards: {eval_episode_rewards_mean}. eval_episode_serve_ratio_mean: {eval_episode_serve_ratio_mean:.4f}. eval_time: {eval_time:.4f}")
                self.logger.info(f"eval_time: {eval_time:.4f}. eval_average_episode_rewards: {eval_episode_rewards_mean:.4f}.")
                self.logger.info(f"eval_episode_serve_stage1_ratio_mean: {eval_episode_serve_stage1_ratio_mean:.4f}. eval_episode_serve_stage2_ratio_mean: {eval_episode_serve_stage2_ratio_mean:.4f}. eval_episode_serve_ratio_mean: {eval_episode_serve_ratio_mean:.4f}.")
                # self.logger.info(f"stage1_rewards_mean: {eval_episode_stage1_rewards_mean:.4f}. stage2_rewards_mean: {eval_episode_stage2_rewards_mean:.4f}. stage3_rewards_mean: {eval_episode_stage3_rewards_mean:.4f}")
                self.logger.info(f"courier_cost_mean: {eval_episode_courier_cost_mean:.4f}. drone_cost_mean: {eval_episode_drone_cost_mean:.4f}. obj_mean: {eval_episode_stage3_rewards_mean:.4f} . profit_mean: {eval_episode_profit_mean:.4f}")
                break
        return eval_episode_rewards_mean
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        
        next_values = self.trainer.policy.get_values(self.buffer.obs[-1], self.buffer.rnn_states_critic[-1])

        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)      
        self.buffer.after_update()
        return train_infos

    def save(self, episode, is_best=False):
        """Save policy's actor and critic networks."""
        self.policy.save(self.save_dir, episode, is_best)

    def restore(self, model_path):
        """Restore policy's networks from a saved model."""
        self.policy.restore(model_path)