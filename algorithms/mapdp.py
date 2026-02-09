import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from .models import BatchNorm, Encoder, MLP, Attention, IterMixin
from envs.mvdpdp.mvdpdp import obs_func_map, trans_func_map, DiscreteMVDPDP


class MAPDP(nn.Module, IterMixin):

    @staticmethod
    def obs_func(env:DiscreteMVDPDP, obs, unobs):
        if env._global["frame"] == 0:
            env.vehicles["start_node"] = env.vehicles["target"].clone()
            env.vehicles["task_target"] = torch.arange(env.n_vehicle, device=env.device).repeat(env.batch_size, 1)
        obs["vehicles"]["task_target"] = env.vehicles["task_target"].clone()
        obs["vehicles"]["start_node"] = env.vehicles["start_node"].clone()
    
    @staticmethod
    def trans_func(env:DiscreteMVDPDP, actions):
        new_task_target = actions["new_task_target"]
        assert ((new_task_target >= 0) & (new_task_target <= env.n_tot_requests * 2 + env.n_vehicle)).all()
        new_task_mask = new_task_target != env.n_tot_requests * 2 + env.n_vehicle
        env.vehicles["task_target"][new_task_mask] = new_task_target[new_task_mask]

    @staticmethod
    def register_algo_specific():
        obs_func_map["mapdp"] = MAPDP.obs_func
        trans_func_map["mapdp"] = MAPDP.trans_func


    def __init__(self, n_enc_bloc, n_embd, n_head, rel_dim, env_args=None):
        super().__init__()
        IterMixin.__init__(self)

        self.n_embd = n_embd
        self.n_head = n_head
        self.rel_dim = rel_dim
        self.env_args = env_args

        # BatchNorm should be turn off when rollout/evaluate
        self.global_bn = BatchNorm(2)
        self.global_proj_in = nn.Linear(2, n_embd)
        self.request_proj_in = nn.Linear(4, n_embd)
        self.request_request_dist_proj_in = MLP(1, n_embd, rel_dim)
        self.value_linear = MLP(n_embd, n_embd, n_embd, 1)
        self.coord_proj_in = nn.Linear(2, n_embd)

        self.depot_proj_in = torch.nn.Linear(n_embd, n_embd)
        self.pickup_proj_in = torch.nn.Linear(n_embd, n_embd)
        self.delivery_proj_in = torch.nn.Linear(n_embd, n_embd)
        self.DEPOT = nn.Parameter(torch.randn(n_embd))
        self.PICKUP = nn.Parameter(torch.randn(n_embd))
        self.DELIVERY = nn.Parameter(torch.randn(n_embd))
        self.NO_TARGET = nn.Parameter(torch.randn(n_embd))
        self.NO_ACTION = nn.Parameter(torch.randn(n_embd))

        self.encoder = Encoder(n_enc_bloc, n_embd, n_head, qkfeat_dim=rel_dim)

        self.proj_h = nn.Linear(env_args["n_vehicle"]*(1+n_embd)+2 * n_embd + 1, n_embd) # large parameter
        self.cross_attn = Attention(n_embd, n_head)
        self.to_q_g = nn.Linear(n_embd, n_embd)
        self.to_k_g = nn.Linear(n_embd, n_embd)


    def forward(self, obs, input_actions=None, deterministic=False, only_critic=False):
        """
        Note: 不要对输入进行 inplace 操作
        Note: max_visible_
        """
        nodes = obs["nodes"]
        vehicles = obs["vehicles"]
        requests = obs["requests"]
        _global = obs["global"]

        batch_size = _global["n_exist_requests"].shape[0]
        n_vehicle = vehicles["capacity"].shape[1]
        n_request = requests["value"].shape[1] # max_consider_requests
        n_node = nodes["n_vehicle"].shape[1]
        device = nodes["n_vehicle"].device
        batch_arange = torch.arange(batch_size, device=device)
        vehicle_arange = torch.arange(n_vehicle, device=device)

        vehicle_decode_mask = vehicles["time_left"] == 0

        global_feature = self.global_proj_in(self.global_bn(torch.stack((
            _global["frame"].to(torch.float),
            _global["n_exist_requests"].to(torch.float),
            # _global["n_unoccur_requests"].to(torch.float),
        ), dim=1)))

        requests_mask = torch.arange(n_request, device=device) < _global["n_consider_requests"][:, None]
        depot_pickup_delivery = torch.cat((vehicles["start_node"], requests["from"].masked_fill(requests_mask, 0), requests["to"].masked_fill(requests_mask, 0)), -1)
        # MAPDP 研究的是2D欧式空间的问题，现在扩展到非欧距离矩阵后不存在坐标，所以无法得到最初表征。
        requests_coord = nodes["coord"].gather(1, requests["from"][:, :, None].expand(-1, -1, 2))
        requests_feature = self.request_proj_in(torch.stack((
            requests["value"].to(torch.float),
            requests["volumn"].to(torch.float),
            requests_coord[:, :, 0].to(torch.float),
            requests_coord[:, :, -1].to(torch.float),
        ), dim=1).transpose(-1, -2))
        depot_nodes_dense = self.DEPOT[None, None].expand(batch_size, n_vehicle, -1)
        pickup_nodes_dense = self.PICKUP[None, None].expand(batch_size, n_request, -1) + requests_feature
        delivery_nodes_dense = self.DELIVERY[None, None].expand(batch_size, n_request, -1) + requests_feature
        depot_pickup_delivery_embd_pre = torch.cat((
            self.depot_proj_in(depot_nodes_dense),
            self.pickup_proj_in(pickup_nodes_dense + delivery_nodes_dense),
            self.delivery_proj_in(delivery_nodes_dense)
        ), -2) + global_feature[:, None, :] + self.coord_proj_in(nodes["coord"].to(torch.float).gather(1, depot_pickup_delivery.unsqueeze(-1).expand(-1, -1, 2)))
        # 加入距离矩阵来捕获节点之间的距离关系
#        dse_dist_mat = _global["node_node"]\
#                .gather(1, depot_pickup_delivery[:, :, None].expand(-1, -1, n_node))\
#                    .gather(2, depot_pickup_delivery[:, None, :].expand(-1, depot_pickup_delivery.shape[-1], -1))
#        rel_mat = self.request_request_dist_proj_in(dse_dist_mat.to(torch.float).unsqueeze(-1))
        rel_mat = None
        depot_pickup_delivery_mask = torch.cat((torch.ones(batch_size, n_vehicle, dtype=torch.bool, device=device), requests_mask, requests_mask), -1)
        depot_pickup_delivery_embd = self.encoder.forward(depot_pickup_delivery_embd_pre, depot_pickup_delivery_mask, rel_mat)

        depot_nodes_embd, pickup_nodes_embd, delivery_nodes_embd = depot_pickup_delivery_embd.split([n_vehicle, n_request, n_request], dim=1)
        # 因为一开始最初表征无意义，所以得到最终表征后再进行一次配对。
        pickup_nodes_embd = pickup_nodes_embd + delivery_nodes_embd
        depot_pickup_delivery_embd = torch.cat((depot_nodes_embd, pickup_nodes_embd, delivery_nodes_embd), 1)
        
        global_embd = torch.cat((depot_nodes_embd, pickup_nodes_embd.masked_fill(requests_mask[..., None], 0), delivery_nodes_embd.masked_fill(requests_mask[..., None], 0)), 1).sum(1) / (1 + requests_mask.sum(-1)[:, None])
        # unweighted version.
        values = self.value_linear(global_embd).squeeze(-1)
        if only_critic:
            return values

        task_target = vehicles["task_target"]
        vehicle_target_task_embd = depot_pickup_delivery_embd[batch_arange[:, None], task_target]
        vehicle_target_task_embd[task_target == -1] = self.NO_TARGET
        
        comm = torch.cat((vehicles["space"], vehicle_target_task_embd.flatten(1)), -1) # (13)
        h = torch.cat((vehicle_target_task_embd, vehicles["space"][:, :, None], global_embd[:, None, :].expand(-1, n_vehicle, -1), comm[:, None, :].expand(-1, n_vehicle, -1)), -1)
        g = self.cross_attn.forward(self.proj_h(h), depot_pickup_delivery_embd, depot_pickup_delivery_embd) # (14)
        q = self.to_q_g(g) # (15)
        k = self.to_k_g(depot_pickup_delivery_embd[:, n_vehicle:]) # (15)
        D = 10 # (16)
        d = self.n_embd / self.n_head # (16)
        k_with_noaction = torch.cat((k, self.NO_ACTION.expand(batch_size, 1, -1)), 1)
        u_with_noaction = D * (q @ k_with_noaction.transpose(-1, -2) / d ** 0.5).tanh() # (16)

        # update vehicle_request status
        cur_vehicle_request = _global["vehicle_request"].clone()
        cur_space = vehicles["space"].clone()
        assert ((cur_vehicle_request == 1).sum(-1) <= 1).all()
        cur_pickup_mask = (vehicles["time_left"] == 0) & ((cur_vehicle_request == 1).sum(-1) == 1)
        pickup_batch, pickup_vehicle = torch.where(cur_pickup_mask)
        pickup_request = task_target[pickup_batch, pickup_vehicle] - n_vehicle
        assert (cur_vehicle_request[pickup_batch, pickup_vehicle, pickup_request] == 1).all()
        assert (vehicles["target"][pickup_batch, pickup_vehicle] == requests["from"][pickup_batch, pickup_request]).all()
        cur_vehicle_request[pickup_batch, pickup_vehicle, pickup_request] = 2
        cur_space[cur_pickup_mask] -= requests["volumn"][pickup_batch, pickup_request]

        # Add pre mask
        mask = torch.ones(batch_size, n_vehicle, n_request * 2, device=device, dtype=torch.bool)
        mask[:, :, :n_request].masked_fill_(~requests_mask[:, None], False)
        mask[:, :, n_request:].masked_fill_(~requests_mask[:, None], False)
        no_relation_mask = (cur_vehicle_request == 0).all(1)
        mask[:, :, :n_request].masked_fill_(~no_relation_mask[:, None], False)
        mask[:, :, n_request:][cur_vehicle_request != 2] = False
        mask[..., :n_request][cur_space[:, :, None] < requests["volumn"][:, None, :]] = False
        mask.masked_fill_(~vehicle_decode_mask[..., None], False)

        u_with_noaction[:, :, :-1].masked_fill_(~mask, -torch.inf)
        u_with_noaction[..., -1].masked_fill_((u_with_noaction[:, :, :-1] == -torch.inf).all(-1), 1)
        p_with_noaction = F.softmax(u_with_noaction, -1) # (17)
        p_with_noaction = p_with_noaction.masked_fill(torch.cat((torch.zeros_like(p_with_noaction, dtype=torch.bool)[..., :-1], (p_with_noaction[..., :-1].sum(-1) != 0).unsqueeze(-1)), dim=-1), 0)

        cate = Categorical(probs=p_with_noaction)

        is_decoding = input_actions is None
        if is_decoding:
            sampled_new_task_target = cate.sample()
            
            # fleet handler, deterministic version.
            # decode new_task_target
            new_task_target = sampled_new_task_target.clone()
            for idx in range(n_vehicle):
                new_pickup_vehicle_mask = vehicle_decode_mask[:, idx] & (new_task_target[:, idx] < n_request)
                new_pickup_batch = batch_arange[new_pickup_vehicle_mask]
                new_pickup_request = new_task_target[:, idx][new_pickup_vehicle_mask]
                no_conflict = (cur_vehicle_request[new_pickup_batch, :, new_pickup_request] == 0).all(-1)
                new_task_target[torch.where(new_pickup_vehicle_mask)[0][~no_conflict], idx] = 2 * n_request
                assert (new_task_target[new_pickup_vehicle_mask, idx][~no_conflict] == 2 * n_request).all()
                cur_vehicle_request[new_pickup_batch[no_conflict], idx, new_pickup_request[no_conflict]] = 1
            
            assert ((cur_vehicle_request == 1).sum(1) <= 1).all()
            assert (new_task_target <= n_request * 2).all()
            pickup_mask = (new_task_target < n_request)
            delivery_mask = (new_task_target >= n_request) & (new_task_target < n_request * 2)
            # decode vehicle
            vehicles_action = vehicles["target"].clone()
            vehicles_action[pickup_mask] = requests["from"][batch_arange[:, None].masked_select(pickup_mask), new_task_target[pickup_mask]]
            vehicles_action[delivery_mask] = requests["to"][batch_arange[:, None].masked_select(delivery_mask), new_task_target[delivery_mask] - n_request]
            # decode request
            requests_action = torch.full([batch_size, n_request], n_vehicle, dtype=torch.int64, device=device)
            requests_action[pickup_batch, pickup_request] = pickup_vehicle
            # decode preassign
            preassign = torch.full([batch_size, n_request], n_vehicle, dtype=torch.int64, device=device)
            preassign[batch_arange[:, None].masked_select(pickup_mask), new_task_target[pickup_mask]] = vehicle_arange.masked_select(pickup_mask)
        else:
            sampled_new_task_target = input_actions["sampled_new_task_target"]
            vehicles_action = input_actions["vehicle"]
            requests_action = input_actions["request"]
            new_task_target = input_actions["new_task_target"]
            preassign = input_actions["preassign"]

        log_prob = cate.log_prob(sampled_new_task_target).sum(-1)
        entropy = cate.entropy().sum(-1)

        # print((new_task_target[0] < n_request).sum())

        return {"vehicle": vehicles_action, "request": requests_action, "preassign": preassign, "sampled_new_task_target": sampled_new_task_target, "new_task_target": new_task_target + n_vehicle}, log_prob, entropy, values

MAPDP.register_algo_specific()