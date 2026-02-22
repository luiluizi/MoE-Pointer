import torch
from utils.util import update_linear_schedule, update_cosine_schedule

class PointerTransformerPolicy:
    def __init__(self, args, env_args, device, only_heuristic=False):
        self.device = device
        self.lr = args.lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.check_grad = args.check_grad
        self.only_heuristic = only_heuristic
        self.algorithm = args.algorithm

        if self.algorithm in ["moe_pointer"]:
            from algorithms.transfer_pointer_transformer_with_heter import MultiAgentPointerTransformer
            print("use transfer_pointer_transformer_with_heter")
            self.transformer = MultiAgentPointerTransformer(
                n_enc_bloc=args.n_enc_block, n_dec_block=args.n_dec_block, n_embd=args.n_embd, n_head=args.n_head, rel_dim=2*args.n_head, env_args=env_args, device=self.device,
                use_ar=not args.not_use_ar, use_relation=not args.not_use_relation,
                use_node_emb=not args.not_use_node_emb, only_heuristic=self.only_heuristic, use_unbind_decode=args.use_unbind_decode, use_moe=args.use_moe,
                hypers={"other_node_prob": args.other_node_prob, "no_assign_prob": args.no_assign_prob, "load_balance_weight": args.load_balance_weight, "temperature": args.temperature}
            )
        elif self.algorithm in ["prob_heuristic"]:
            from algorithms.transfer_pointer_transformer2 import MultiAgentPointerTransformer
            print("use transfer_pointer_transformer2")
            self.transformer = MultiAgentPointerTransformer(
                n_enc_bloc=args.n_enc_block, n_dec_block=args.n_dec_block, n_embd=args.n_embd, n_head=args.n_head, rel_dim=2*args.n_head, env_args=env_args, device=self.device,
                use_ar=not args.not_use_ar, use_heur_req=not args.not_use_heur_req, use_heur_veh=not args.not_use_heur_veh, use_relation=not args.not_use_relation, use_nearest_station = args.use_nearest_station,
                use_node_emb=not args.not_use_node_emb, only_heuristic=self.only_heuristic, use_unbind_decode=args.use_unbind_decode, use_moe=args.use_moe,
                hypers={"other_node_prob": args.other_node_prob, "no_assign_prob": args.no_assign_prob, "load_balance_weight": args.load_balance_weight, "temperature": args.temperature}
            )
        elif self.algorithm == "mapdp":
            from algorithms.MAPDP import MAPDP
            self.transformer = MAPDP(
                n_enc_bloc=args.n_enc_block, n_embd=args.n_embd, n_head=args.n_head, rel_dim=2*args.n_head, env_args=env_args
            )

        # count the volume of parameters of model
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        for param in self.transformer.parameters():
            mulValue = param.numel()
            Total_params += mulValue
            if param.requires_grad:
                Trainable_params += mulValue
            else:
                NonTrainable_params += mulValue
        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=self.lr, 
                                          eps=self.opti_eps,
                                          weight_decay=self.weight_decay)
        self.transformer.to(self.device)

        if self.check_grad:
            torch.autograd.set_detect_anomaly(True)

    
    def lr_warmup(self, episode, warmup_episode):
        lr = (episode + 1) / (warmup_episode + 1) * self.lr 
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def lr_origin(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        
    def lr_decay(self, episode, episodes, option):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        if option == 'linear':
            update_linear_schedule(self.optimizer, episode, episodes, self.lr)
        elif option == 'cosine':
            update_cosine_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, obs, rnn_states_actor, rnn_states_critic, deterministic=False):
        actions, action_log_probs, entropy, values = self.transformer.forward(obs, None, deterministic=deterministic)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, obs, rnn_states_critic):
        return self.transformer.forward(obs, None, only_critic=True)

    def evaluate_actions(self, obs, rnn_states_actor, rnn_states_critic, actions):
        actions, action_log_probs, entropy, values = self.transformer.forward(obs, actions)
        return values, action_log_probs, entropy
 
    def act(self, obs, rnn_states_actor, deterministic=True):
        actions, action_log_probs, entropy, values = self.transformer.forward(obs, None, deterministic=deterministic)
        return actions, rnn_states_actor

    def save(self, save_dir, episode, is_best=False):
        self.transformer.iter.fill_(episode)
        if is_best:
            torch.save(self.transformer.state_dict(), str(save_dir) + f"/transformer_best.pt")
        else:
            torch.save(self.transformer.state_dict(), str(save_dir) + f"/transformer_{episode}.pt")

    def restore(self, model_path):
        transformer_state_dict = torch.load(model_path, weights_only=True)
        if hasattr(self.transformer, "iter") and "iter" not in transformer_state_dict:
            transformer_state_dict["iter"] = torch.tensor(0)
        if not hasattr(self.transformer, "iter") and "iter" in transformer_state_dict:
            del transformer_state_dict["iter"]
        self.transformer.load_state_dict(transformer_state_dict)

    def train(self):
        self.transformer.train()

    def eval(self):
        self.transformer.eval()