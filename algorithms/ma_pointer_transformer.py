"""
rel     relation
acc     accumulate
rep     representation
mat     matrix
cur     current
"""

from itertools import product, permutations

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Categorical

from .models import BatchNorm, BatchNorm2d, Encoder, Decoder, MLP, assign_symmetric, IterMixin, MulConstant, Identity, NoEmbedding


class MultiAgentPointerTransformer(nn.Module, IterMixin):

    def __init__(self, n_enc_bloc, n_dec_block, n_embd, n_head, rel_dim, max_len=1100, env_args=None, use_ar=True, use_heur_req=True, use_heur_veh=True, use_relation=True, use_tsp=True, use_node_emb=False, use_unbind_decode=False, only_heuristic=False, hypers=None):
        super().__init__()
        IterMixin.__init__(self)

        self.n_embd = n_embd
        self.n_head = n_head
        self.rel_dim = rel_dim
        self.env_args = env_args

        # options
        # TODO: 为什么不启用 softmask 会导致 reward 如此之低。
        self.use_heur_req = use_heur_req
        self.use_heur_veh = use_heur_veh
        self.use_ar = use_ar
        self.only_heuristic = only_heuristic
        self.use_relation = use_relation
        self.use_tsp = use_tsp
        self.use_node_emb = use_node_emb
        self.use_unbind_decode = use_unbind_decode
        assert not self.use_unbind_decode or use_ar

        # BatchNorm should be turn off when rollout/evaluate
        self.node_bn = BatchNorm(3)
        self.node_proj_in = nn.Linear(3, n_embd)
        self.vehicle_bn = BatchNorm(3)
        self.vehicle_proj_in = nn.Linear(3, n_embd)
        self.request_bn = BatchNorm(2)
        self.request_proj_in = nn.Linear(2, n_embd)
        self.global_bn = BatchNorm(2)
        self.global_proj_in = nn.Linear(2, n_embd)
        self.node_node_bn = BatchNorm2d(1)
        self.request_request_dist_bn = BatchNorm2d(4)

        if self.use_ar:
            self.pos_embd = nn.Embedding(max_len, n_embd)
            # self.pos_embd.weight.data *= 0.5
        else:
            self.pos_embd = NoEmbedding(n_embd)

        self.node_node_proj_in = MLP(1, n_embd, rel_dim, scale=4., enable_scale=True)
        self.vehicle_node_embd = nn.Sequential(nn.Embedding(2, n_embd), MLP(n_embd, rel_dim, pre_act=True, scale=12., enable_scale=True))
        self.request_node_embd = nn.Sequential(nn.Embedding(3, n_embd), MLP(n_embd, rel_dim, pre_act=True, scale=15., enable_scale=True))
        self.vehicle_request_embd = nn.Sequential(nn.Embedding(4, n_embd), MLP(n_embd, rel_dim, pre_act=True, scale=8., enable_scale=True))
        self.request_request_dist_proj_in = MLP(4, n_embd, rel_dim, scale=8., enable_scale=True)
        
        self.rep_action_proj_in = nn.Linear(n_embd * 2, n_embd)
        self.value_linear = nn.Linear(n_embd, 1)

        self.SOS = nn.Parameter(torch.randn(n_embd)) # should be init to correct scale?
        self.NOACTION = nn.Parameter(torch.randn(n_embd))
        self.node_emb = nn.Parameter(torch.randn(500, n_embd))
        if self.use_node_emb:
            self.register_buffer("stored_use_node_emb", torch.tensor(self.use_node_emb))

        if not self.only_heuristic:
            self.encoder = Encoder(n_enc_bloc, n_embd, n_head, qkfeat_dim=rel_dim)
            self.decoder = Decoder(n_dec_block, n_embd, n_head, qkfeat_dim=rel_dim)
        else:
            self.encoder = Identity()
            self.decoder = Identity()

        # Buffers
        # only use for eval/inference
        self.perm = torch.tensor(list(permutations(list(range(self.env_args["max_capacity"])))), device="cuda:0")
        
        # Tunable Hyper Parameters
        # Affact train reward, but has less effect on eval reward.
        self.other_node_prob = hypers.get("other_node_prob", 0.00) # i.g. stay inplace or goto a node where no request. # \beta_2
        self.no_assign_prob = hypers.get("no_assign_prob", 0.03) # \beta_1
        self.load_balance_weight = hypers.get("load_balance_weight", 0.5) # \alpha_1
        self.concentrat_weight = hypers.get("concentrat_weight", 0.5) # \alpha_2

        self.dist_norm = 3.0 if env_args["scenario"] == "synthetic-large2" else 1.0

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.perm = self.perm.to(self.SOS.device)

    # TODO:
    # Enable Mixed Precision Training
    # Enable Distributed Training
    # @torch.autocast("cuda", torch.float16)
    # @torch.compile
    def forward(self, obs, input_actions=None, deterministic=False, only_critic=False, temperature=1.0):
        """
        Note: 不要对输入进行 inplace 操作

        TODO: Sparse Attention mask
        TODO: mask 作为引导/监督吗？
        soft_mask: 训练采样时用，但是不覆盖原 logits。事实证明，不太行。
        TODO: hard_mask: 训练采样时用，eval 时也用，覆盖原 logits。
        非法信息(padded requests)需要在 softmax 前mask
        合法信息但非法操作（满载，无需决策）需要在 softmax 后mask
        先解码 requests 再解码 vehicle
        Profile 显存占用。broadcast 矩阵乘法特别慢，而且占显存特别多。
        TODO: 上模仿学习了。
        TODO: probs != 0 assertion
        TODO: torch.compile Attention
        TODO: 改为非 AR 训练
        TODO: scale up
        """
        nodes = obs["nodes"]
        vehicles = obs["vehicles"]
        requests = obs["requests"]
        _global = obs["global"]
        batch_size = _global["n_exist_requests"].shape[0]
        device = nodes["n_vehicle"].device

        n_node = nodes["n_vehicle"].shape[1]
        n_vehicle = vehicles["capacity"].shape[1]
        n_request = requests["value"].shape[1] # max_consider_requests
        node_arange = torch.arange(n_node, device=device)

        dist_label = _global["node_node"].to(torch.float32).sum(-1).sum(-1) / (n_node * n_node) * 0.7 # heuristic

        nodes_feature = self.node_proj_in(self.node_bn(torch.stack((
            nodes["n_vehicle"].to(torch.float),
            nodes["n_requests_from"].to(torch.float),
            nodes["n_requests_to"].to(torch.float)
        ), dim=1)).transpose(-1, -2))
        if self.stored_use_node_emb:
            nodes_feature = nodes_feature + self.node_emb[:n_node]

        vehicles_feature = self.vehicle_proj_in(self.vehicle_bn(torch.stack((
            vehicles["capacity"].to(torch.float),
            vehicles["space"].to(torch.float),
            vehicles["time_left"].to(torch.float)
        ), dim=1)).transpose(-1, -2))

        requests_feature = self.request_proj_in(self.request_bn(torch.stack((
            requests["value"].to(torch.float),
            requests["volumn"].to(torch.float)
        ), dim=1)).transpose(-1, -2))

        global_feature = self.global_proj_in(self.global_bn(torch.stack((
            _global["frame"].to(torch.float),
            _global["n_exist_requests"].to(torch.float),
            # _global["n_unoccur_requests"].to(torch.float),
        ), dim=1)))

        lens = [n_node, n_vehicle, n_request, 1]
        acc_lens = np.cumsum(lens)
        rel_mat = torch.zeros(batch_size, sum(lens), sum(lens), self.rel_dim, device=device)
        all_mask = torch.ones(batch_size, sum(lens), dtype=torch.bool, device=device)
        all_mask[:, sum(lens[:2]):sum(lens[:3])][torch.arange(lens[2], device=device) >= _global["n_consider_requests"][:, None]] = False
        requests_mask = torch.arange(lens[2], device=device) < _global["n_consider_requests"][:, None]
        # node - node
        node_node_feature = self.node_node_proj_in(self.node_node_bn(_global["node_node"][:, None].to(torch.float).div(self.dist_norm))[:, 0][..., None])
        rel_mat[:, :n_node, :n_node] = node_node_feature

        # node - vehicle
        batch_arange = torch.arange(batch_size, device=device)
        vehicle_arange = torch.arange(n_vehicle, device=device)
        vehicle_node_token = torch.zeros(batch_size, n_vehicle, n_node, dtype=torch.int64, device=device)
        vehicle_node_token[batch_arange[:, None], vehicle_arange, vehicles["target"]] = 1
        vehicle_node_feature = self.vehicle_node_embd(vehicle_node_token)
        assign_symmetric(rel_mat, slice(acc_lens[0], acc_lens[1]), slice(acc_lens[0]), vehicle_node_feature)

        # node - request
        request_arange = torch.arange(n_request, device=device)
        request_node_token = torch.zeros(batch_size, n_request, n_node, dtype=torch.int64, device=device)
        request_node_token[batch_arange[:, None].expand(-1, n_request)[requests_mask], request_arange.expand(batch_size, -1)[requests_mask], requests["from"][requests_mask]] = 1
        request_node_token[batch_arange[:, None].expand(-1, n_request)[requests_mask], request_arange.expand(batch_size, -1)[requests_mask], requests["to"][requests_mask]] = 2
        request_node_feature = self.request_node_embd(request_node_token)
        assign_symmetric(rel_mat, slice(acc_lens[1], acc_lens[2]), slice(acc_lens[0]), request_node_feature)

        # vehicle - request
        vehicle_request_token = _global["vehicle_request"].masked_fill(~requests_mask[:, None, :].expand(-1, n_vehicle, -1), 0)
        vehicle_request = self.vehicle_request_embd(vehicle_request_token)
        assign_symmetric(rel_mat, slice(acc_lens[0], acc_lens[1]), slice(acc_lens[1], acc_lens[2]), vehicle_request)

        # TODO: request - request
        request_request_feature = torch.stack([
            _global["node_node"].div(self.dist_norm)\
                .gather(1, requests[key1].masked_fill(~requests_mask, 0)[:, :, None].expand(-1, -1, n_node))\
                    .gather(2, requests[key2].masked_fill(~requests_mask, 0)[:, None, :].expand(-1, n_request, -1))\
                        for key1, key2 in product(["from", "to"], ["from", "to"])
        ], -1)
        rel_mat[:, acc_lens[1]:acc_lens[2], acc_lens[1]:acc_lens[2]] = self.request_request_dist_proj_in(self.request_request_dist_bn(request_request_feature.to(torch.float).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))

        if not self.use_relation:
            rel_mat = torch.zeros_like(rel_mat)


        # global counter
        # if "counter" in globals():
        #     counter += 1
        # else: 
        #     counter = 0
        
        # print(node_node_feature.abs().mean(),
        #       vehicle_node_feature.abs().mean(),
        #       request_node_feature.abs()mean(),
        #       vehicle_request.abs().mean(),
        #       self.request_request_dist_proj_in(request_request_feature.to(torch.float)).abs().mean())
        # rep: representation
        # 一层卷积无法做 request - vehicle 之间通过 node 交互，必须多层才行。
        all_rep = self.encoder.forward(torch.cat((
            nodes_feature, vehicles_feature, requests_feature, global_feature.unsqueeze(-2)
        ), dim=-2) + self.pos_embd(torch.arange(acc_lens[-1], device=device)[None]), all_mask, rel_mat if self.use_relation else None)

        # all_rep = torch.zeros_like(all_rep) # for debug usage.
        nodes_rep, vehicles_rep, requests_rep, global_rep = all_rep.split(lens, dim=1)

        values = self.value_linear(global_rep.squeeze(-2)).squeeze(-1)
        if only_critic:
            return values
        # because *_feature has less infomation, so we use *_rep(resentation) as decoder target
        # Decoder(vehicles[->node] + requests[->vehicle])
        # 因为 causal 解码，先解码不带 mask 的 vehicle。
        # 当前解码逻辑有点奇怪，因为解码出的操作对应输入到下一个Agent中，但是应该能捕获到与本Agent的关系？
        # RNN 可以work，是因为RNN默认有序列关系。Transformer如果不加位置编码，仅依靠causal解码，虽然能捕获Agent与action对应关系，但是我们也加了 Positional Embedding 来加强捕获。
        # 方案1：action 与 representation concat
        # 方案2：action 与 representation 分别占一个位置

        log_prob = torch.zeros(batch_size, device=device)
        entropy = torch.zeros(batch_size, device=device)
        is_decoding = input_actions is None
        if is_decoding:
            # TODO: 将不需要解码的直接赋值
            vehicles_action = vehicles["target"].clone()
            requests_action = torch.full([batch_size, n_request], n_vehicle, dtype=torch.int64, device=device)
        else:
            vehicles_action = input_actions["vehicle"]
            requests_action = input_actions["request"]

        # request node to
        request_node_from = torch.zeros(batch_size, n_request, n_node, dtype=torch.bool, device=device)
        request_node_to = torch.zeros(batch_size, n_request, n_node, dtype=torch.bool, device=device)
        request_node_from[batch_arange[:, None].masked_select(requests_mask), request_arange.masked_select(requests_mask), requests["from"][requests_mask]] = True
        request_node_to[batch_arange[:, None].masked_select(requests_mask), request_arange.masked_select(requests_mask), requests["to"][requests_mask]] = True


        # shit usage.
        if self.use_tsp and  _global["frame"].max().item() / self.env_args["n_frame"] > 0.8:
            chunk_size = min(8 if self.env_args["max_capacity"] == 6 else 128, batch_size)
            base_overlength_mask_list = []
            for i in range(0, batch_size, chunk_size):
                tsp_batch, tsp_vehicle, tsp_request = torch.where(_global["vehicle_request"][i:i+chunk_size]== 2)
                tsp_index = (_global["vehicle_request"][i:i+chunk_size] == 2).cumsum(-1)[tsp_batch, tsp_vehicle, tsp_request] - 1
                tsp_points = requests["to"][i:i+chunk_size, ..., None, None].repeat(1, 1, n_vehicle, self.env_args["max_capacity"])
                tsp_points[tsp_batch, :, tsp_vehicle, tsp_index] = requests["to"][i:i+chunk_size][tsp_batch, tsp_request, None] # B x K x c
                tsp_routes_ = tsp_points[:, :, :, None, :].expand(-1, -1, -1, self.perm.shape[0], -1).gather(-1, self.perm[None, None, None, :, :].expand(min(chunk_size, batch_size-i), n_request, n_vehicle, -1, -1)) # B x K x P(c) x c
                tsp_routes = torch.cat((vehicles["target"][i:i+chunk_size, None, :, None, None].expand(-1, n_request, -1, self.perm.shape[0], -1), tsp_routes_), dim=-1)
                tsp_edges = _global["node_node"][batch_arange[i:i+chunk_size, None, None, None, None], tsp_routes[..., :-1], tsp_routes[..., 1:]]
                vehicle_finish_length = tsp_edges.sum(-1).min(-1).values # B x K

                base_overlength_mask = vehicle_finish_length > (self.env_args["n_frame"] - _global["frame"])[i:i+chunk_size, None, None]
                base_overlength_mask = base_overlength_mask.transpose(-1, -2)
                base_overlength_mask_list.append(base_overlength_mask)
            base_overlength_mask = torch.cat(base_overlength_mask_list, dim=0)
        else:
            base_overlength_mask = torch.zeros(batch_size, n_vehicle, n_request, dtype=torch.bool, device=device)
        

        #################### Decoding Request Actions. ####################
        if self.use_ar:
            last_action_rep = self.SOS.expand(batch_size, -1)
        else:
            last_action_rep = torch.zeros(batch_size, self.SOS.shape[-1], device=device)

        # _vehicle_node_have_request_to = ((_global["vehicle_request"] == 2).unsqueeze(-1) & request_node_to.unsqueeze(1)).any(-2)
        # _vehicle_node_have_request_to[batch_arange[:, None], vehicle_arange, vehicles["target"]] = True
        cur_spaces = vehicles["space"].clone() # don't mutate input tensors.
        cur_vehicle_request = _global["vehicle_request"].clone()

        vehicles_rep_with_nopickup = torch.cat((vehicles_rep, self.NOACTION.expand(batch_size, 1, self.NOACTION.shape[-1])), dim=-2)
        all_requests_decode_idx = torch.empty(batch_size, 0, dtype=torch.int64, device=device)
        all_requests_decode_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=device)
        # all_requests_decode_count = 0
        # all_requests_noaction_count = 0
        for _idx in range(n_request):
            vehicle_request_pickup_mask = (cur_vehicle_request==0).all(-2, keepdim=True) & (vehicles["time_left"] == 0)[:, :, None] & (vehicles["target"][:, :, None] == requests["from"][:, None, :])
            requests_can_decode_mask = (vehicle_request_pickup_mask & (cur_spaces[:, :, None] >= requests["volumn"][:, None, :]) & ~base_overlength_mask).any(-2)

            requests_decode_idx = requests_can_decode_mask.to(torch.int64).argmax(-1)
            requests_decode_mask = requests_can_decode_mask[batch_arange, requests_decode_idx]
            if (~requests_decode_mask).all():
                break

            all_requests_decode_idx = torch.cat((all_requests_decode_idx, requests_decode_idx[:, None]), dim=1)
            all_requests_decode_mask = torch.cat((all_requests_decode_mask, requests_decode_mask[:, None]), dim=1)

            if self.use_unbind_decode:
                input_emb = requests_rep[batch_arange, requests_decode_idx]
            else:
                input_emb = self.rep_action_proj_in(torch.cat((requests_rep[batch_arange, requests_decode_idx], last_action_rep), dim=-1))
                # print(requests_rep[batch_arange, requests_decode_idx].abs().mean(), last_action_rep.abs().mean(), self.pos_embd(requests_decode_idx).abs().mean(), input_emb.abs().mean())
            dec_rel_mat = rel_mat[batch_arange, n_node+n_vehicle+requests_decode_idx].gather(1, n_node+n_vehicle+all_requests_decode_idx[..., None].expand(-1, -1, self.rel_dim))[:, None]
            no_ar_mask = torch.zeros(batch_size, _idx+1, dtype=torch.bool, device=device); no_ar_mask[..., -1] = True; # we use no_ar_mask to mask previous tokens, which equivalent to input separatly.
            pointer_hidden = self.decoder.forward(
                (input_emb + self.pos_embd(requests_decode_idx)).unsqueeze(1), None if self.use_ar else no_ar_mask,
                None if not self.use_relation else dec_rel_mat if not self.use_unbind_decode else dec_rel_mat.repeat_interleave(2, dim=-2)[:, :, :-1],
                all_rep, all_mask, rel_mat[batch_arange, n_node+n_vehicle+requests_decode_idx][:, None] if self.use_relation else None, use_kvcache=True
            ).squeeze(1)
            action_logits = (pointer_hidden.unsqueeze(-2) @ vehicles_rep_with_nopickup.transpose(-1, -2)).squeeze(-2)
            if temperature != 1.0:
                action_logits /= temperature
            action_probs = torch.nn.functional.softmax(action_logits, -1) + 1e-5 # NOTE: add 1e-5 avoiding nan
            if self.only_heuristic:
                action_probs = torch.ones_like(action_probs)
            
            action_probs = action_probs.masked_fill(F.pad(cur_spaces<requests["volumn"][batch_arange, requests_decode_idx, None], (0, 1), "constant", False), 0.)
            # 如果是padded request/ delivering request，这里使用mask直接让其选择最后一个，其采样概率为1，并且梯度为0。
            action_probs = action_probs.masked_fill(F.pad(~vehicle_request_pickup_mask[batch_arange, :, requests_decode_idx], (0, 1), "constant", False), 0.)

            # TSP heuristic, maybe no bugs.
            if self.use_tsp:
                tsp_batch, tsp_vehicle, tsp_request = torch.where((cur_vehicle_request if self.use_ar else _global["vehicle_request"])== 2)
                tsp_index = ((cur_vehicle_request if self.use_ar else _global["vehicle_request"]) == 2).cumsum(-1)[tsp_batch, tsp_vehicle, tsp_request] - 1
                tsp_points = requests["to"][batch_arange, requests_decode_idx, None, None].repeat(1, n_vehicle, self.env_args["max_capacity"])
                tsp_points[tsp_batch, tsp_vehicle, tsp_index] = requests["to"][tsp_batch, tsp_request] # B x K x c
                tsp_routes_ = tsp_points[:, :, None, :].expand(-1, -1, self.perm.shape[0], -1).gather(-1, self.perm[None, None, :, :].expand(batch_size, n_vehicle, -1, -1)) # B x K x P(c) x c
                tsp_routes = torch.cat((vehicles["target"][:, :, None, None].expand(-1, -1, self.perm.shape[0], -1), tsp_routes_), dim=-1)
                tsp_edges = _global["node_node"][batch_arange[:, None, None, None], tsp_routes[..., :-1], tsp_routes[..., 1:]]
                vehicle_finish_length = tsp_edges.sum(-1).min(-1).values # B x K

                _tsp_points = vehicles["target"][:, :, None].repeat(1, 1, self.env_args["max_capacity"])
                _tsp_points[tsp_batch, tsp_vehicle, tsp_index] = requests["to"][tsp_batch, tsp_request] # B x K x c
                _tsp_routes_ = _tsp_points[:, :, None, :].expand(-1, -1, self.perm.shape[0], -1).gather(-1, self.perm[None, None, :, :].expand(batch_size, n_vehicle, -1, -1)) # B x K x P(c) x c
                _tsp_routes = torch.cat((vehicles["target"][:, :, None, None].expand(-1, -1, self.perm.shape[0], -1), _tsp_routes_), dim=-1)
                _tsp_edges = _global["node_node"][batch_arange[:, None, None, None], _tsp_routes[..., :-1], _tsp_routes[..., 1:]]
                _vehicle_finish_length = _tsp_edges.sum(-1).min(-1).values # B x K

                delivery_dist_increment = vehicle_finish_length - _vehicle_finish_length
                overlength_mask = vehicle_finish_length > (self.env_args["n_frame"] - _global["frame"])[:, None]
                assert (delivery_dist_increment >= 0).all()
                action_probs = action_probs.masked_fill(torch.nn.functional.pad(overlength_mask, (0, 1), "constant", False), 0.)
            else:
                delivery_dist_increment = torch.zeros(batch_size, n_vehicle, device=device)
                
            if self.use_heur_req:
                
                # 负载均衡
                softmask_loadbalance_probs = cur_spaces / self.env_args["max_capacity"] * self.load_balance_weight * 0.5
                # 计算 request_to_diversity，其为如果当前 request 加入到某个 vehicle 里，其到达该 vehicle 其它 request_to 的最小距离
                # to_node = requests["to"][batch_arange, requests_decode_idx].masked_fill(~requests_mask[batch_arange, requests_decode_idx], 0)
                # delivery_dist_increment = torch.min(
                #     _global["node_node"][batch_arange, None, to_node, :].masked_fill(~_vehicle_node_have_request_to, torch.iinfo(torch.int64).max).min(-1).values,
                #     _global["node_node"][batch_arange, None, :, to_node].masked_fill(~_vehicle_node_have_request_to, torch.iinfo(torch.int64).max).min(-1).values
                # ) # B x n_vehicle
                # 对于 0 request，其 to_diversity 为 0
                delivery_dist_increment.masked_fill(cur_spaces == vehicles["capacity"], 0)
                # assert (delivery_dist_increment != torch.iinfo(torch.int64).max).all()
                # TODO: 为什么与距离有关，不应该啊，该改。
                softmask_delivery_diversity_probs = dist_label[:, None] / (delivery_dist_increment + 1) * self.concentrat_weight # TODO: check heuristic value.
                # env_uniform: (softmask_loadbalance_probs + softmask_delivery_diversity_probs).min() > 0.25
                # import ipdb;ipdb.set_trace()
                softmask_probs_with_no_action = torch.nn.functional.pad(softmask_loadbalance_probs + softmask_delivery_diversity_probs, (0, 1), "constant", self.no_assign_prob)
                action_probs = action_probs * softmask_probs_with_no_action
                # action_probs = MulConstant.apply(action_probs, softmask_probs_with_no_action)

            action_probs = action_probs.masked_fill(F.pad(~requests_decode_mask[:, None].expand(-1, n_vehicle), (0, 1), "constant", False), 0.)

            # dangerous
            no_action_mask = (action_probs == .0).all(-1)
            action_probs[no_action_mask, -1] = 1.0

            cate = Categorical(probs=action_probs)
            if is_decoding:
                if deterministic:
                    action = cate.probs.argmax(dim=-1)
                else:
                    action = cate.sample()
                requests_action[batch_arange[requests_decode_mask], requests_decode_idx[requests_decode_mask]] = action[requests_decode_mask]
            else:
                action = requests_action[batch_arange, requests_decode_idx]
                action[~requests_decode_mask] = n_vehicle
            
            # all_requests_decode_count += (cate.probs[requests_decode_mask][:, -1] < 0.999).sum().item()
            # all_requests_noaction_count += (action[requests_decode_mask & (cate.probs[:, -1] < 0.999)] == n_vehicle).sum()

            action_log_prob = cate.log_prob(action)
            log_prob[requests_decode_mask] += action_log_prob[requests_decode_mask] # padded requests 强制使其只有一种决策而不让其产生梯度。
            entropy[requests_decode_mask] += cate.entropy()[requests_decode_mask]

            assign_to_vehicle_mask = action != n_vehicle
            if self.use_ar:
                # update states
                cur_spaces[batch_arange[assign_to_vehicle_mask], action[assign_to_vehicle_mask]] -= requests["volumn"][batch_arange[assign_to_vehicle_mask], requests_decode_idx[assign_to_vehicle_mask]]
                assert (assign_to_vehicle_mask <= requests_decode_mask).all()
                assert (cur_spaces >= 0).all()

            if self.use_ar:
                last_action_rep = last_action_rep.masked_scatter(requests_decode_mask[:, None].expand(-1, self.n_embd), vehicles_rep_with_nopickup[batch_arange[requests_decode_mask], action[requests_decode_mask]])
            # _vehicle_node_have_request_to[batch_arange[assign_to_vehicle_mask], action[assign_to_vehicle_mask], requests["to"][batch_arange, requests_decode_idx][assign_to_vehicle_mask]] = True
            cur_vehicle_request[batch_arange[assign_to_vehicle_mask], action[assign_to_vehicle_mask], requests_decode_idx[assign_to_vehicle_mask]] = 2
            # 如果不标记，可能导致不断解码这个东西。
            cur_vehicle_request[batch_arange[requests_decode_mask&~assign_to_vehicle_mask], :, requests_decode_idx[requests_decode_mask&~assign_to_vehicle_mask]] = 5

            if self.use_unbind_decode:
                pointer_hidden = self.decoder.forward(
                    (last_action_rep + self.pos_embd(requests_decode_idx)).unsqueeze(1), None,
                    dec_rel_mat.repeat_interleave(2, dim=-2) if self.use_relation else None,
                    all_rep, all_mask, rel_mat[batch_arange, n_node+n_vehicle+requests_decode_idx][:, None] if self.use_relation else None, use_kvcache=True
                ).squeeze(1)
        # print(f"decodeing {_idx} requests action, frame {_global['frame'][0]}")
        # 经过验证，在 tw 数据集上，是可能出现 all_requests_noaction_count > 0 的情况的
        # if all_requests_decode_count > 0:
        #     print(all_requests_noaction_count / all_requests_decode_count)

        #################### Decoding Vehicle Actions. ####################
        # TODO: 这里解码为原 target 还是 NOACTION
        # Note: kmask for padded requests.
        # TODO: only decode offway vehicles.
        # NOACTION 用处太多了，是不是应该把 SOS 和 NOACTION 拆开
        all_vehicles_decode_idx = torch.empty(batch_size, 0, dtype=torch.int64, device=device)
        cur_vehicle_need_decode = vehicles["time_left"] == 0
        if not self.use_ar:
            cur_vehicle_request = _global["vehicle_request"].clone()
        for _idx in range(n_vehicle):
            vehicle_decode_idx = cur_vehicle_need_decode.to(torch.int64).argmax(-1)
            vehicle_decode_mask = cur_vehicle_need_decode[batch_arange, vehicle_decode_idx]
            if (~vehicle_decode_mask).all():
                break

            if not self.use_ar:
                # conflict handler
                pickuping_mask = requests_action == vehicle_decode_idx[:, None]
                overvolumn_mask = ((pickuping_mask * requests["volumn"][batch_arange[:, None], requests_action.clamp_max(n_vehicle - 1)]).cumsum(-1) > vehicles["space"][batch_arange, vehicle_decode_idx][:, None]) * pickuping_mask
                requests_action[overvolumn_mask] = n_vehicle
                # print(overvolumn_mask.sum())

            all_vehicles_decode_idx = torch.cat((all_vehicles_decode_idx, vehicle_decode_idx[:, None]), dim=1)

            if self.use_unbind_decode:
                input_emb = vehicles_rep[batch_arange, vehicle_decode_idx]
            else:
                input_emb = self.rep_action_proj_in(torch.cat((vehicles_rep[batch_arange, vehicle_decode_idx], last_action_rep), dim=-1))
                # print(vehicles_rep[batch_arange, vehicle_decode_idx].abs().mean(), last_action_rep.abs().mean(), self.pos_embd(vehicle_decode_idx).abs().mean(), input_emb.abs().mean())
            dec_rel_mat = torch.cat((
                rel_mat[batch_arange, n_node+vehicle_decode_idx].gather(1, n_node+n_vehicle+all_requests_decode_idx[..., None].expand(-1, -1, self.rel_dim))[:, None],
                rel_mat[batch_arange, n_node+vehicle_decode_idx].gather(1, n_node+all_vehicles_decode_idx[..., None].expand(-1, -1, self.rel_dim))[:, None],
            ), dim=2)
            dec_mask = F.pad(all_requests_decode_mask, (0, _idx+1), "constant", True)
            no_ar_mask = dec_mask.clone(); no_ar_mask[..., -1] = True
            pointer_hidden = self.decoder.forward(
                (input_emb + self.pos_embd(vehicle_decode_idx)).unsqueeze(1),
                no_ar_mask if not self.use_ar else dec_mask if not self.use_unbind_decode else dec_mask.repeat_interleave(2, dim=1)[:, :-1],
                None if not self.use_relation else dec_rel_mat if not self.use_unbind_decode else dec_rel_mat.repeat_interleave(2, dim=-2)[:, :, :-1],
                all_rep, all_mask, rel_mat[batch_arange, n_node+vehicle_decode_idx][:, None] if self.use_relation else None, use_kvcache=True
            ).squeeze(1)
            action_logits = (pointer_hidden.unsqueeze(-2) @ nodes_rep.transpose(-1, -2)).squeeze(-2)
            if temperature != 1.0:
                action_logits /= temperature
            action_probs = torch.nn.functional.softmax(action_logits, -1) + 1e-5
            if self.only_heuristic:
                action_probs = torch.ones_like(action_probs)
            # TODO: 解码时只能去往存在 request 的地方或者自己携带 request 的目的地。
            # time_left_maskout = torch.zeros_like(action_probs, dtype=torch.bool)
            # time_left_maskout[(vehicles["time_left"][:, idx]!=0)[:, None] & (vehicles["target"][:, idx, None] != node_arange)] = True
            # action_probs = action_probs.masked_fill(time_left_maskout, 0.)

            # softmask guided exploration.
            # 带时间窗口的 heuristic
            # 其它点，概率为 0.2
            # request_from，概率按照距离排序 max_dist * 0.3 / dist，要考虑是否满载，Reasonable?
            node_have_reqeust_from = ((cur_vehicle_request[batch_arange, vehicle_decode_idx] == 0).unsqueeze(-1) & request_node_from).any(-2)
            distance_prob = (dist_label[:, None] /_global["node_node"][batch_arange, vehicles["target"][batch_arange, vehicle_decode_idx]]).clamp_max(1.) * 0.1
            assert distance_prob.isfinite().all()
            if not self.use_heur_veh:
                distance_prob = torch.ones_like(distance_prob)
            # 如果有剩余空间，并且该点有物品，那么可以去，否则基本不可能去。
            # distance_prob <= 0.1
            softmask_probs = torch.where((vehicles["space"][batch_arange, vehicle_decode_idx] != 0)[:, None] & node_have_reqeust_from, distance_prob, self.other_node_prob) # Tunable Hyper Parameter.
            # print((softmask_probs == self.other_node_prob).sum()/batch_size)
            # softmask_probs = torch.ones_like(softmask_probs) * self.other_node_prob
            # request_to，概率为 1。当然是有可能和 request_from 重复的。
            node_have_reqeust_to = ((cur_vehicle_request[batch_arange, vehicle_decode_idx] == 2).unsqueeze(-1) & request_node_to).any(-2)
            softmask_probs[node_have_reqeust_to] = 1.0 # Fuck, 这个值必须远大于 distance_prob 和 self.other_node_prob
            action_probs = action_probs * softmask_probs # Add or Multiply.
            # action_probs = MulConstant.apply(action_probs, softmask_probs)

            if self.use_ar and self.use_tsp:
                tsp_batch, tsp_request = torch.where(cur_vehicle_request[batch_arange, vehicle_decode_idx] == 2)
                tsp_index = (cur_vehicle_request[batch_arange, vehicle_decode_idx] == 2).cumsum(-1)[tsp_batch, tsp_request] - 1
                tsp_points = node_arange[None, :, None].repeat(batch_size, 1, self.env_args["max_capacity"])
                tsp_points[tsp_batch, :, tsp_index] = requests["to"][tsp_batch, tsp_request][:, None] # B x N x capacity
                tsp_routes_ = tsp_points[:, :,  None, :].repeat(1, 1, self.perm.shape[0], 1).gather(-1, self.perm[None, None, :, :].repeat(batch_size, n_node, 1, 1)) # B x P(capacity) x capacity
                # B x N x P(capacity) x (capacity + 1)
                tsp_routes = torch.cat((node_arange[None, :, None, None].repeat(batch_size, 1, self.perm.shape[0], 1), tsp_routes_), dim=-1)
                # B x N x P(capacity) x capacity
                tsp_edges = _global["node_node"][batch_arange[:, None, None, None], tsp_routes[..., :-1], tsp_routes[..., 1:]]
                node_finish_length = tsp_edges.sum(-1).min(-1).values + _global["node_node"][batch_arange[:, None], vehicles["target"][batch_arange, vehicle_decode_idx, None], node_arange[None, :]]
                # corner case, stay local.
                node_finish_length[batch_arange, vehicles["target"][batch_arange, vehicle_decode_idx]] += 1
                overlength_mask = node_finish_length > (self.env_args["n_frame"] - _global["frame"])[:, None]
                action_probs[overlength_mask] = .0

            # dangerous
            # 没有合法动作就待在原地。
            no_action_mask = (action_probs == .0).all(-1)
            action_probs[batch_arange[no_action_mask], vehicles["target"][batch_arange, vehicle_decode_idx][no_action_mask]] = 1.0

            # 只要 logits 不是全 -inf 就可以
            cate = Categorical(probs=action_probs)
            if is_decoding:
                if deterministic:
                    action = cate.probs.argmax(dim=-1)
                else:
                    action = cate.sample()
                vehicles_action[batch_arange[vehicle_decode_mask], vehicle_decode_idx[vehicle_decode_mask]] = action[vehicle_decode_mask]
            else:
                action = vehicles_action[batch_arange, vehicle_decode_idx]
            # TODO: masked_add
            action_log_prob = cate.log_prob(action)
            log_prob[vehicle_decode_mask] += action_log_prob[vehicle_decode_mask]
            entropy[vehicle_decode_mask] += cate.entropy()[vehicle_decode_mask]
            if self.use_ar:
                last_action_rep = nodes_rep[batch_arange, action]
            cur_vehicle_need_decode[batch_arange[vehicle_decode_mask], vehicle_decode_idx[vehicle_decode_mask]] = False

            if self.use_unbind_decode:
                pointer_hidden = self.decoder.forward(
                    (last_action_rep + self.pos_embd(vehicle_decode_idx)).unsqueeze(1),
                    dec_mask.repeat_interleave(2, dim=1),
                    dec_rel_mat.repeat_interleave(2, dim=-2) if self.use_relation else None,
                    all_rep, all_mask, rel_mat[batch_arange, n_node+vehicle_decode_idx][:, None] if self.use_relation else None, use_kvcache=True
                ).squeeze(1)

        self.decoder.reset_kvcache()

        return {"vehicle": vehicles_action, "request": requests_action}, log_prob, entropy, values