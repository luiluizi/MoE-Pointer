import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.valuenorm import ValueNorm
from algorithms.pointer_transformer_policy import PointerTransformerPolicy


class MATTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):
        self.all_args = args

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy: PointerTransformerPolicy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.mini_batch_size = args.mini_batch_size
        self.value_loss_coef = args.value_loss_coef
        self.aux_loss_coef = args.aux_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = not args.not_use_valuenorm
        
        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks and not self.dec_actor:
        value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
        adv_targ = sample
        
        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            obs_batch, 
            rnn_states_batch, 
            rnn_states_critic_batch, 
            actions_batch
        )
        # actor update
        # TODO: 验证 surrogate loss 稳定性, 在 not_use_heur_req 时, imp_weights 非常非常小。
        # TODO: 验证有多少有效的项没被 clip 掉。
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        if self.all_args.not_use_ar: # which causes training process much unstable.
            imp_weights = imp_weights.clamp_max(1e5)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        policy_loss = -torch.min(surr1, surr2)
        policy_loss = policy_loss.masked_fill(~policy_loss.isfinite(), 0).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)

        # entropy loss
        dist_entropy = dist_entropy.mean()
        # aux_losses
        # aux_losses = aux_losses.mean()
        # loss = - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef
        # loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef + aux_losses * self.aux_loss_coef 
        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef
        
        # print("value_loss", value_loss)
        # print("dist_entropy", dist_entropy)
        # print("policy_loss", policy_loss)

        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_gard_norm(self.policy.transformer.parameters())
        
        
        # for p in self.policy.transformer.parameters():
        #     if p.grad is not None and p.grad.isnan().any():
        #         import ipdb
        #         ipdb.set_trace()
        any_nan = False
        for name, p in self.policy.transformer.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                # print(f"Found NaNs in grad of parameter: {name}, shape={tuple(p.shape)}")
                any_nan = True
        assert not any_nan
        self.policy.optimizer.step()

        return value_loss, grad_norm, policy_loss, dist_entropy, grad_norm, imp_weights

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        num_updates = 0
        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(self.mini_batch_size)
            id = 1
            for sample in data_generator:
                # print(id)
                id += 1
                # import pdb
                # pdb.set_trace()
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                
                num_updates += 1

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
