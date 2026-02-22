import tree
import torch
from envs.env import DroneTransferEnv

class ReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, env, device):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        # RNN
        self.recurrent_N = args.recurrent_N
        self.recurrent_embd = args.n_embd
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = not args.not_use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = not args.not_use_valuenorm
        self.algo = args.algorithm
        self.device = device
        self.env: DroneTransferEnv = env

        self.obs = [None] * (self.episode_length + 1)

        self.rnn_states = torch.zeros(self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.recurrent_embd, device=self.device)
        self.rnn_states_critic = torch.zeros_like(self.rnn_states)

        self.value_preds = torch.zeros(self.episode_length + 1, self.n_rollout_threads, device=self.device)
        self.returns = torch.zeros_like(self.value_preds)
        self.advantages = torch.zeros(self.episode_length, self.n_rollout_threads, device=self.device)

        self.actions = [None] * self.episode_length

        # 存储每个时间步采取某动作时的log概率
        self.action_log_probs = torch.zeros(self.episode_length, self.n_rollout_threads, device=self.device)
        self.rewards = torch.zeros(self.episode_length, self.n_rollout_threads, device=self.device)

        self.masks = torch.ones(self.episode_length + 1, self.n_rollout_threads, dtype=torch.bool, device=self.device)

        self.step = 0

    def insert(self, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.obs[self.step + 1] = obs
        self.rnn_states[self.step + 1] = rnn_states_actor
        self.rnn_states_critic[self.step + 1] = rnn_states_critic
        self.actions[self.step] = actions
        self.action_log_probs[self.step] = action_log_probs
        self.value_preds[self.step] = value_preds
        self.rewards[self.step] = rewards
        self.masks[self.step + 1] = masks
        # 达到episode_length后重置
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        #在策略/价值网络更新结束后，需要让 buffer 保留最后一个 time step 的状态，作为下一轮数据收集的初始状态。
        """Copy last timestep data to first index. Called after update to model."""
        self.obs[0] = self.obs[-1]
        self.rnn_states[0] = self.rnn_states[-1]
        self.rnn_states_critic[0] = self.rnn_states_critic[-1]
        self.masks[0] = self.masks[-1]

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        # next_value从哪来

        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                    self.value_preds[step + 1]) * self.masks[step + 1] \
                        - value_normalizer.denormalize(self.value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                # if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                #     gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            else:
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                        self.masks[step + 1] - self.value_preds[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                # dose this patch useful?
                # if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                #     gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + self.value_preds[step]
        
        # TODO: standardization advantages
        # TODO: Advatange Normalization should done in group.
        # TODO: get statistics of advantages.
        # NOTE: this is correct if batch_size is large.
        if self.advantages.shape[1] > 1:
            mean_advantages = self.advantages.mean(1, keepdim=True)
            std_advantages = self.advantages.std(1, keepdim=True)
            self.std_advantages = (self.advantages - mean_advantages) / (std_advantages + 1e-5)
        else:
            # only use for batch_size = 1
            self.std_advantages = self.advantages.clone()

    def feed_forward_generator_transformer(self, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        """
        episode_length, n_rollout_threads = self.rewards.shape
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size > batch_size:
            mini_batch_size = batch_size
        if batch_size % mini_batch_size != 0:
            print("warning, batch_size % mini_batch_size != 0, use droplast")
        num_mini_batch = batch_size // mini_batch_size

        rand = torch.randperm(batch_size, device=self.env.device)
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        # we don't use agent shuffle.
        batched_obs = tree.map_structure(lambda *args: torch.cat(args) if isinstance(args[0], torch.Tensor) else None, *self.obs)
        batched_rnn_states = self.rnn_states[:-1].flatten(end_dim=1)
        batched_rnn_states_critic = self.rnn_states_critic[:-1].flatten(end_dim=1)
        batched_action = tree.map_structure(lambda *args: torch.cat(args), *self.actions)
        batched_value_preds = self.value_preds[:-1].flatten(end_dim=1)
        batched_returns = self.returns.flatten(end_dim=1)
        batched_mask = self.masks[:-1].flatten(end_dim=1)
        batched_action_log_probs = self.action_log_probs.flatten(end_dim=1)
        batched_advantages = self.std_advantages.flatten(end_dim=1)
        for indices in sampler:
            obs_batch = tree.map_structure(lambda t: t[indices] if isinstance(t, torch.Tensor) else None, batched_obs)
            rnn_states_batch = batched_rnn_states[indices]
            rnn_states_critic_batch = batched_rnn_states_critic[indices]
            actions_batch = tree.map_structure(lambda t: t[indices], batched_action)
            value_preds_batch = batched_value_preds[indices]
            return_batch = batched_returns[indices]
            masks_batch = batched_mask[indices]
            old_action_log_probs_batch = batched_action_log_probs[indices]
            adv_targ = batched_advantages[indices]

            yield obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                  adv_targ
