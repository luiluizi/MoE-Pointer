from itertools import product, permutations
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Categorical

from .component.models import BatchNorm, Encoder, Decoder, MLP, assign_symmetric, IterMixin, MulConstant, Identity, NoEmbedding, TOKEN_REQ_ASSIGN, TOKEN_COURIER_NEXT, TOKEN_DRONE_NEXT
from .myModels import PointerWithPrior


class MultiAgentPointerTransformer(nn.Module, IterMixin):

    def __init__(self, n_enc_bloc, n_dec_block, n_embd, n_head, rel_dim, device, max_len=1100, env_args=None, use_ar=True, use_heur_req=True, use_heur_veh=True, use_relation=True, use_node_emb=False, use_unbind_decode=False, use_nearest_station = False, only_heuristic=False, use_moe=False, hypers=None):
        super().__init__()
        IterMixin.__init__(self)
        self.n_embd = n_embd
        self.n_head = n_head
        self.rel_dim = rel_dim
        self.env_args = env_args
        self.device = device

        self.use_heur_req = use_heur_req
        self.use_heur_veh = use_heur_veh
        self.use_ar = use_ar
        self.only_heuristic = only_heuristic
        self.use_relation = use_relation
        self.use_node_emb = use_node_emb
        self.use_unbind_decode = use_unbind_decode
        self.use_nearest_station = use_nearest_station
        self.use_moe = use_moe
        assert not self.use_unbind_decode or use_ar
        self.top = 2

        # BatchNorm should be turn off when rollout/evaluate
        self.node_bn = BatchNorm(4)
        self.node_proj_in = nn.Linear(4, n_embd)
        self.courier_bn = BatchNorm(3)
        self.courier_proj_in = nn.Linear(3, n_embd)
        self.drone_bn = BatchNorm(1)
        self.drone_proj_in = nn.Linear(1, n_embd)
        self.request_bn = BatchNorm(2)
        self.request_proj_in = nn.Linear(2, n_embd)
        
        self.global_bn = BatchNorm(2)
        self.global_proj_in = nn.Linear(2, n_embd)
        
        self.node_node_bn = BatchNorm(1)
        self.request_request_dist_bn = BatchNorm(4)

        if self.use_ar:
            self.pos_embd = nn.Embedding(max_len, n_embd)
        else:
            self.pos_embd = NoEmbedding(n_embd)

        self.node_node_proj_in = MLP(1, n_embd, rel_dim, scale=4., enable_scale=True)
        # For Courier (Vehicle)
        self.vehicle_node_embd = nn.Sequential(nn.Embedding(2, n_embd), MLP(n_embd, rel_dim, pre_act=True, scale=12., enable_scale=True))
        self.vehicle_request_embd = nn.Sequential(nn.Embedding(4, n_embd), MLP(n_embd, rel_dim, pre_act=True, scale=8., enable_scale=True))
        
        # For Drone (New)
        self.drone_node_embd = nn.Sequential(nn.Embedding(2, n_embd), MLP(n_embd, rel_dim, pre_act=True, scale=12., enable_scale=True))
        self.drone_request_embd = nn.Sequential(nn.Embedding(4, n_embd), MLP(n_embd, rel_dim, pre_act=True, scale=8., enable_scale=True))

        # Request Node Embedding (Updated to 5 to include stations)
        self.request_node_embd = nn.Sequential(nn.Embedding(5, n_embd), MLP(n_embd, rel_dim, pre_act=True, scale=15., enable_scale=True))
        
        self.request_request_dist_proj_in = MLP(4, n_embd, rel_dim, scale=8., enable_scale=True)
        
        self.rep_action_proj_in = nn.Linear(n_embd * 2, n_embd)
        self.value_linear = nn.Linear(n_embd, 1)
        self.courier_pointer = PointerWithPrior(n_embd)
        
        self.SOS = nn.Parameter(torch.randn(n_embd)) # should be init to correct scale?
        self.NOACTION = nn.Parameter(torch.randn(n_embd))
        
        # Entity Type Embeddings (可学习的实体类型嵌入)
        # 使用nn.Embedding来创建可学习的实体类型嵌入
        self.entity_type_emb = nn.Embedding(4, n_embd)  # 0: node, 1: courier, 2: drone, 3: request

        self.node_emb = nn.Parameter(torch.randn(500, n_embd))
        if self.use_node_emb:
            self.register_buffer("stored_use_node_emb", torch.tensor(self.use_node_emb))

        if not self.only_heuristic:
            # self.encoder = Identity()
            # print("use Identity encoder")
            self.encoder = Encoder(n_enc_bloc, n_embd, n_head, rel_dim)
            self.decoder = Decoder(n_dec_block, n_embd, n_head, rel_dim, use_moe=use_moe)
        else:
            self.encoder = Identity()
            self.decoder = Identity()

        # Buffers
        # only use for eval/inference
        # self.perm = torch.tensor(list(permutations(list(range(self.env_args["max_capacity"])))), device=self.device)
        
        # Tunable Hyper Parameters
        # Affact train reward, but has less effect on eval reward.
        self.other_node_prob = hypers.get("other_node_prob", 0.00) # i.g. stay inplace or goto a node where no request. # \beta_2
        self.no_assign_prob = hypers.get("no_assign_prob", 0.00) # \beta_1
        self.load_balance_weight = hypers.get("load_balance_weight", 0.5) # \alpha_1
        self.concentrat_weight = hypers.get("concentrat_weight", 0.5) # \alpha_2
        self.temperature = hypers.get("temperature", 2.0) # \alpha_2
        
        self.dist_norm = 1.0 # Default dist norm

    def forward(self, obs, input_actions=None, deterministic=False, only_critic=False):  
        nodes = obs["nodes"]
        drones = obs["drones"]
        couriers = obs["couriers"]
        requests = obs["requests"]
        _global = obs["global"]
        batch_size = _global["n_exist_requests"].shape[0]
        device = nodes["n_courier"].device

        n_node = nodes["n_courier"].shape[1]
        n_station = _global["station_idx"].shape[1] - 1
        n_courier = couriers["capacity"].shape[1]
        n_drone = drones["target"].shape[1]
        n_request = requests["value"].shape[1] # max_consider_requests
        node_arange = torch.arange(n_node, device=device)

        # 平均距离
        dist_label = _global["node_node"].to(torch.float32).sum(-1).sum(-1) / (n_node * n_node)  # heuristic 0.35
        dist_label_drone = _global["station_node_node"].to(torch.float32).sum(-1).sum(-1) / (n_station * n_station) * 0.1
        
        nodes_feature = self.node_proj_in(self.node_bn(torch.stack((
            nodes["n_courier"].to(torch.float),
            nodes["n_drone"].to(torch.float),
            nodes["n_requests_from"].to(torch.float),
            nodes["n_requests_to"].to(torch.float),
        ), dim=1)).transpose(-1, -2))
        
        couriers_feature = self.courier_proj_in(self.courier_bn(torch.stack((
            couriers["capacity"].to(torch.float),
            couriers["space"].to(torch.float),
            couriers["time_left"].to(torch.float)
        ), dim=1)).transpose(-1, -2))

        drones_feature = self.drone_proj_in(self.drone_bn(
            drones["time_left"].to(torch.float).unsqueeze(-1)))

        requests_feature = self.request_proj_in(self.request_bn(torch.stack((
            requests["value"].to(torch.float),
            requests["volumn"].to(torch.float)
        ), dim=1)).transpose(-1, -2))

        global_feature = self.global_proj_in(self.global_bn(torch.stack((
            _global["frame"].to(torch.float),
            _global["n_exist_requests"].to(torch.float),
        ), dim=1)))
        
        # nodes_feature = nodes_feature + self.node_type_emb
        # couriers_feature = couriers_feature + self.courier_type_emb
        # drones_feature = drones_feature + self.drone_type_emb
        # requests_feature = requests_feature + self.request_type_emb
        
        # 放在 forward 开头、在计算 lens 之前或之后立刻打印
        batch_size = _global["n_exist_requests"].shape[0]
        device = nodes["n_courier"].device

        n_node = nodes["n_courier"].shape[1]
        n_courier = couriers["capacity"].shape[1]
        n_drone = drones["target"].shape[1]
        n_request = requests["value"].shape[1]
        
        lens = [n_node, n_courier, n_drone, n_request, 1]
        acc_lens = np.cumsum(lens)

        rel_mat = torch.zeros(batch_size, sum(lens), sum(lens), self.rel_dim, device=device)
        # mask掉未consider的request
        all_mask = torch.ones(batch_size, sum(lens), dtype=torch.bool, device=device)
        all_mask[:, sum(lens[:3]):sum(lens[:4])][torch.arange(lens[3], device=device) >= _global["n_consider_requests"][:, None]] = False
        requests_mask = torch.arange(lens[3], device=device) < _global["n_consider_requests"][:, None]
        batch_arange = torch.arange(batch_size, device=device)
        courier_arange = torch.arange(n_courier, device=device)
        drone_arrange = torch.arange(n_drone, device=device)
        request_arange = torch.arange(n_request, device=device)

        # ---------------- Rel Mat Construction ----------------
        # node - node
        node_node_feature = self.node_node_proj_in(self.node_node_bn(_global["node_node"][:, None].to(torch.float).div(self.dist_norm))[:, 0][..., None])
        rel_mat[:, :n_node, :n_node] = node_node_feature

        # courier - node
        courier_node_token = torch.zeros(batch_size, n_courier, n_node, dtype=torch.int64, device=device)
        courier_node_token[batch_arange[:, None], courier_arange, couriers["target"]] = 1
        courier_node_feature = self.vehicle_node_embd(courier_node_token)
        assign_symmetric(rel_mat, slice(acc_lens[0], acc_lens[1]), slice(acc_lens[0]), courier_node_feature)

        # drone - node
        drone_node_token = torch.zeros(batch_size, n_drone, n_node, dtype=torch.int64, device=device)
        drone_node_token[batch_arange[:, None], drone_arrange, drones["target"]] = 1
        drone_node_feature = self.drone_node_embd(drone_node_token)
        assign_symmetric(rel_mat, slice(acc_lens[1], acc_lens[2]), slice(acc_lens[0]), drone_node_feature)

        # request - node
        request_node_token = torch.zeros(batch_size, n_request, n_node, dtype=torch.int64, device=device)
        # from = 1
        request_node_token[batch_arange[:, None].expand(-1, n_request)[requests_mask], request_arange.expand(batch_size, -1)[requests_mask], requests["from"][requests_mask]] = 1
        # to = 2
        request_node_token[batch_arange[:, None].expand(-1, n_request)[requests_mask], request_arange.expand(batch_size, -1)[requests_mask], requests["to"][requests_mask]] = 2
        
        # station1 = 3 (if assigned)
        station1_mask = requests_mask & (requests["station1"] != n_node)
        request_node_token[batch_arange[:, None].expand(-1, n_request)[station1_mask], request_arange.expand(batch_size, -1)[station1_mask], requests["station1"][station1_mask]] = 3
        
        # station2 = 4 (if assigned)
        station2_mask = requests_mask & (requests["station2"] != n_node)
        request_node_token[batch_arange[:, None].expand(-1, n_request)[station2_mask], request_arange.expand(batch_size, -1)[station2_mask], requests["station2"][station2_mask]] = 4

        request_node_feature = self.request_node_embd(request_node_token)
        assign_symmetric(rel_mat, slice(acc_lens[2], acc_lens[3]), slice(acc_lens[0]), request_node_feature)

        # courier - request
        courier_request_token = _global["courier_request"].masked_fill(~requests_mask[:, None, :].expand(-1, n_courier, -1), 0)
        # Clamp values to valid embedding range [0, 3]
        courier_request_token = courier_request_token.clamp(0, 3)
        courier_request = self.vehicle_request_embd(courier_request_token)
        assign_symmetric(rel_mat, slice(acc_lens[0], acc_lens[1]), slice(acc_lens[2], acc_lens[3]), courier_request)

        # drone - request
        drone_request_token = _global["drone_request"].masked_fill(~requests_mask[:, None, :].expand(-1, n_drone, -1), 0)
        # Clamp values to valid embedding range [0, 3]
        drone_request_token = drone_request_token.clamp(0, 3)
        drone_request = self.drone_request_embd(drone_request_token)
        assign_symmetric(rel_mat, slice(acc_lens[1], acc_lens[2]), slice(acc_lens[2], acc_lens[3]), drone_request)

        # request - request
        request_request_feature = torch.stack([
            _global["node_node"].div(self.dist_norm)\
                .gather(1, requests[key1].masked_fill(~requests_mask, 0)[:, :, None].expand(-1, -1, n_node))\
                    .gather(2, requests[key2].masked_fill(~requests_mask, 0)[:, None, :].expand(-1, n_request, -1))\
                        for key1, key2 in product(["from", "to"], ["from", "to"])
        ], -1)
        rel_mat[:, acc_lens[2]:acc_lens[3], acc_lens[2]:acc_lens[3]] = self.request_request_dist_proj_in(self.request_request_dist_bn(request_request_feature.to(torch.float).permute(0, 3, 1, 2)).permute(0, 2, 3, 1))

        if not self.use_relation:
            rel_mat = torch.zeros_like(rel_mat)

        all_rep = self.encoder.forward(torch.cat((
            nodes_feature, 
            couriers_feature, 
            drones_feature,
            requests_feature, 
            global_feature.unsqueeze(-2)
        ), dim=-2) + self.pos_embd(torch.arange(acc_lens[-1], device=device)[None]), all_mask, rel_mat)
        
        if torch.isnan(all_rep).any():
            assert(False)

        nodes_rep, couriers_rep, drones_rep, requests_rep, global_rep = all_rep.split(lens, dim=1)
        
        # TODO 起降站是否作为节点呢
        # 提取起降站特征
        mask_ext = _global["station_mask"].unsqueeze(-1).expand(-1, -1, nodes_rep.size(2))
        flat = nodes_rep.masked_select(mask_ext)
        stations_rep = flat.view(nodes_rep.size(0), -1, nodes_rep.size(2))

        # 得到value
        values = self.value_linear(global_rep.squeeze(-2)).squeeze(-1)
        if only_critic:
            return values

        log_prob = torch.zeros(batch_size, device=device)
        entropy = torch.zeros(batch_size, device=device)
        is_decoding = input_actions is None
        if is_decoding:
            station1_action = torch.full([batch_size, n_request], n_node, dtype=torch.int64, device=device)
            station2_action = torch.full([batch_size, n_request], n_node, dtype=torch.int64, device=device)
            couriers_action = couriers["target"].clone()
            drones_action = drones["target"].clone()
            requests_couriers_action = torch.full([batch_size, n_request], n_courier, dtype=torch.int64, device=device)
            requests_drones_action = torch.full([batch_size, n_request], n_drone, dtype=torch.int64, device=device)
        else:
            station1_action = input_actions["station1"]
            station2_action = input_actions["station2"]
            couriers_action = input_actions["courier"]
            drones_action = input_actions["drone"]
            requests_couriers_action = input_actions["request_courier"]
            requests_drones_action = input_actions["request_drone"]

        #################### Decoding Request Actions. ####################
        # 是否要拼接上一步动作的表征
        if self.use_ar:
            last_action_rep = self.SOS.expand(batch_size, -1)
        else:
            last_action_rep = torch.zeros(batch_size, self.SOS.shape[-1], device=device)

        cur_spaces_courier = couriers["space"].clone() # don't mutate input tensors.
        cur_spaces_drone = drones["space"].clone()
        cur_courier_request = _global["courier_request"].clone()
        cur_drone_request = _global["drone_request"].clone()

        all_requests_decode_idx = torch.empty(batch_size, 0, dtype=torch.int64, device=device)
        all_requests_decode_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=device)
        
        request_node_from = torch.zeros(batch_size, n_request, n_node, dtype=torch.bool, device=device)
        request_node_to = torch.zeros(batch_size, n_request, n_node, dtype=torch.bool, device=device)
        request_node_station1 = torch.zeros(batch_size, n_request, n_node, dtype=torch.bool, device=device)
        request_node_station2 = torch.zeros(batch_size, n_request, n_node, dtype=torch.bool, device=device)
        request_node_from[batch_arange[:, None].masked_select(requests_mask), request_arange.masked_select(requests_mask), requests["from"][requests_mask]] = True
        request_node_to[batch_arange[:, None].masked_select(requests_mask), request_arange.masked_select(requests_mask), requests["to"][requests_mask]] = True
        
        requests_mask_no_assign_station1 = (requests["station1"] != n_node) & requests_mask
        requests_mask_no_assign_station2 = (requests["station2"] != n_node) & requests_mask
        
        request_node_station1[batch_arange[:, None].masked_select(requests_mask_no_assign_station1), request_arange.masked_select(requests_mask_no_assign_station1), requests["station1"][requests_mask_no_assign_station1]] = True
        request_node_station2[batch_arange[:, None].masked_select(requests_mask_no_assign_station2), request_arange.masked_select(requests_mask_no_assign_station2), requests["station2"][requests_mask_no_assign_station2]] = True
        
        couriers_rep_with_nopickup = torch.cat((couriers_rep, self.NOACTION.expand(batch_size, 1, self.NOACTION.shape[-1])), dim=-2)
        
        # 处理第一三阶段的订单
        for _idx in range(n_request):
            # 判断request处于哪一阶段，然后分别进行处理
            requests_type = torch.zeros(batch_size, n_request, dtype=torch.int64, device=device)
            # 第一阶段
            courier_request_pickup_stage1_mask = (cur_courier_request == 0).all(-2, keepdim=True) & \
                (couriers["time_left"] == 0)[:, :, None] & (couriers["target"][:, :, None] == requests["from"][:, None, :])
            requests_stage1_mask = (courier_request_pickup_stage1_mask & (cur_spaces_courier[:, :, None] >= requests["volumn"][:, None, :])).any(-2)
            requests_type[requests_stage1_mask] = 1
            # 第三阶段 最后一项是为了防止同一个courier完成了一个订单的阶段一和阶段三配送 数据合理的情况下一定是低效的
            courier_request_pickup_stage3_mask = (cur_drone_request==4).any(-2, keepdim=True) & ((cur_courier_request!=5).all(-2, keepdim=True)) & ((cur_courier_request!=2).all(-2, keepdim=True)) & \
                (couriers["time_left"] == 0)[:, :, None] & (couriers["target"][:, :, None] == requests["station2"][:, None, :]) & (cur_courier_request!=3)
            requests_stage3_mask = (courier_request_pickup_stage3_mask & (cur_spaces_courier[:, :, None] >= requests["volumn"][:, None, :])).any(-2)
            requests_type[requests_stage3_mask] = 3

            # [B, n_tot_requests]
            requests_can_decode_mask = requests_stage1_mask | requests_stage3_mask
            # 第一个可以解码的请求
            requests_decode_idx = requests_can_decode_mask.to(torch.int64).argmax(-1)

            # [B]要解码的请求种类
            all_requests_type = requests_type[batch_arange, requests_decode_idx]
            requests_decode_mask = requests_can_decode_mask[batch_arange, requests_decode_idx]
            if (~requests_decode_mask).all():
                break
            # 记录每个batch已经完成解码的请求id
            all_requests_decode_idx = torch.cat((all_requests_decode_idx, requests_decode_idx[:, None]), dim=1) # [B, decoded requests]
            all_requests_decode_mask = torch.cat((all_requests_decode_mask, requests_decode_mask[:, None]), dim=1) # [B, decoded requests]

            # 是否要拼接上一轮的action嵌入与当前的request嵌入作为输入的嵌入
            if self.use_unbind_decode:
                input_emb = requests_rep[batch_arange, requests_decode_idx]
            else:
                input_emb = self.rep_action_proj_in(torch.cat((requests_rep[batch_arange, requests_decode_idx], last_action_rep), dim=-1))
            
            # When use_ar=False, each decoder call is independent, so no_ar_mask should match current input length (1)
            if not self.use_ar:
                self.decoder.reset_kvcache()
            no_ar_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)  # Only mask current token when use_ar=False 
            
            pointer_hidden = self.decoder.forward(
                (input_emb + self.pos_embd(requests_decode_idx)).unsqueeze(1), 
                dec_mask=None if self.use_ar else no_ar_mask,
                rel_mat=None, # DO NOT introduce rel_mat in decoder
                rep_enc=all_rep, 
                enc_mask=all_mask,
                enc_rel_mat=None,
                use_kvcache=self.use_ar,
                token_type=TOKEN_REQ_ASSIGN
            ).squeeze(1)

            # 针对不同类型请求进行解码
            mask1 = all_requests_type == 1
            mask3 = all_requests_type == 3
            mask_courier = mask1 | mask3
            
            assert not (mask1 & mask3).all()

            courier_action_temp = torch.full([batch_size], n_courier, device=device)
            
            # 对应了batch中请求的种类
            if mask1.any() or mask3.any():
                # pointer_hidden [B, n_embed]
                # couriers_rep_with_nopickup [B, n_courier + 1, n_embed]
                action_logits = (pointer_hidden.unsqueeze(-2) @ couriers_rep_with_nopickup.transpose(-1, -2)).squeeze(-2)
                action_probs = F.softmax(action_logits, -1) + 1e-5 # NOTE: add 1e-5 avoiding nan 
                if self.only_heuristic:
                    action_probs = torch.ones_like(action_probs)
                
                # 为其action选择中添加一列，设置其为False, 表示不采取动作
                # 如果当前courier没有剩余空间，设置对应位置为True，使用masked_fill如果为True设置选择该courier的概率为0
                action_probs[batch_arange[mask_courier]] = action_probs[batch_arange[mask_courier]].masked_fill( 
                    F.pad(cur_spaces_courier[batch_arange[mask_courier]] < requests["volumn"][batch_arange[mask_courier], requests_decode_idx[mask_courier], None], (0, 1), "constant", False), 0.)

                # 同上，不在取货点或者不空闲，屏蔽掉
                if mask1.any():
                    action_probs[batch_arange[mask1]] = action_probs[batch_arange[mask1]].masked_fill(
                        F.pad(~courier_request_pickup_stage1_mask[batch_arange[mask1], :, requests_decode_idx[mask1]], (0, 1), "constant", False), 0.)
                if mask3.any():
                    action_probs[batch_arange[mask3]] = action_probs[batch_arange[mask3]].masked_fill(
                        F.pad(~courier_request_pickup_stage3_mask[batch_arange[mask3], :, requests_decode_idx[mask3]], (0, 1), "constant", False), 0.)
                 
                # 启发式概率
                # 负载平衡 courier当前负载越小、运载订单越少，选中的概率越高
                balance_probs = cur_spaces_courier / couriers["capacity"] * self.load_balance_weight * 0.5
                additional_probs = F.pad(balance_probs, (0,1), "constant", self.no_assign_prob)
                
                # 结合启发式概率与模型计算动作概率
                action_probs = action_probs * additional_probs
                # 如果选择所有courier的概率均为0，设置其最后一位的概率为1，也即不被courier取走
                no_action_mask = (action_probs == .0).all(-1)
                action_probs[no_action_mask, -1] = 1.0
          
                cate = Categorical(probs=action_probs)

                if is_decoding:
                    if deterministic:
                        action = cate.probs.argmax(dim=-1)
                    else:
                        action = cate.sample()
                    requests_couriers_action[batch_arange[mask_courier], requests_decode_idx[mask_courier]] = action[mask_courier]
                else:
                    action = requests_couriers_action[batch_arange, requests_decode_idx]
                    action[~requests_decode_mask] = n_courier
                
                courier_action_temp = action
                action_log_prob = cate.log_prob(action)
                log_prob[mask_courier] += action_log_prob[mask_courier] # padded requests 强制使其只有一种决策而不让其产生梯度。
                entropy[mask_courier] += cate.entropy()[mask_courier]

                if self.use_ar:
                    last_action_rep = last_action_rep.masked_scatter(mask_courier[:, None].expand(-1, self.n_embd), 
                                                                    couriers_rep_with_nopickup[batch_arange[mask_courier], action[mask_courier]])

            assign_to_courier_mask = (courier_action_temp != n_courier) & mask_courier
            
            if self.use_ar:
                # 更新载具容量
                cur_spaces_courier[batch_arange[assign_to_courier_mask], courier_action_temp[assign_to_courier_mask]] -= requests["volumn"][batch_arange[assign_to_courier_mask], requests_decode_idx[assign_to_courier_mask]]
                # 可decode才能assign
                assert (assign_to_courier_mask[mask_courier] <= requests_decode_mask[mask_courier]).all()
                assert (cur_spaces_courier >= 0).all()
            # 更新相互之间状态
            # 2表示当前载具搭载该请求， 7表示当前暂时不搭载请求（skip掉）
            cur_courier_request[batch_arange[assign_to_courier_mask], courier_action_temp[assign_to_courier_mask], requests_decode_idx[assign_to_courier_mask]] = 2
            cur_courier_request[batch_arange[requests_decode_mask & ~assign_to_courier_mask], :, requests_decode_idx[requests_decode_mask & ~assign_to_courier_mask]] = 7
        
        # TODO 对于第二阶段不需要决策选择哪个无人机,只要有满足条件的就搭乘
        drone_request_pickup_mask = (cur_courier_request == 3).any(-2, keepdim=True) & (cur_drone_request == 0).all(-2, keepdim=True) & \
            (drones["time_left"] == 0)[:, :, None] & (drones["target"][:, :, None] == requests["station1"][:, None, :]) # [B, n_drone, n_request]
        can_pickup_by_drone  = (drone_request_pickup_mask & (cur_spaces_drone[:, :, None] >= requests["volumn"][:, None, :])) # [B, n_drone, n_request]
        # [B, n_drone, n_request] 给不可用的订单分配大索引值
        request_indices = torch.arange(n_request, device=device).unsqueeze(0).unsqueeze(0).expand(batch_size, n_drone, -1)
        drone_for_request = torch.full((batch_size, n_request), n_drone, dtype=torch.int64, device=device)
        assigned_requests = torch.zeros(batch_size, n_request, dtype=torch.bool, device=device)
        
        for d in range(n_drone):
            available_mask = can_pickup_by_drone[:, d, :] & (~assigned_requests) # [B, n_request] 当前无人机的可用订单（排除已分配的）
            available_indices = torch.where(available_mask, request_indices[:, d, :], torch.tensor(n_request, device=device))
            selected_requests = available_indices.min(dim=-1).values  # [B] 每个batch中，该无人机选择的订单索引（最小可用序号）
            has_assignment = (selected_requests < n_request) # [B] 该无人机是否成功分配到订单
            # 更新分配记录
            drone_for_request[batch_arange[has_assignment], selected_requests[has_assignment]] = d
            assigned_requests[batch_arange[has_assignment], selected_requests[has_assignment]] = True

        is_assigned = (drone_for_request < n_drone) # [B, n_request] 每个订单是否被分配
        requests_drones_action[is_assigned] = drone_for_request[is_assigned] # 更新 requests_drones_action
        
        # 更新 cur_drone_request [B, n_drone, n_request]
        # 构建分配矩阵 [B, n_drone, n_request]
        assignment_matrix = (drone_for_request.unsqueeze(1) == torch.arange(n_drone, device=device).view(1, -1, 1))
        cur_drone_request[assignment_matrix] = 2
        
        # 更新 cur_spaces_drone [B, n_drone]
        # for d in range(n_drone):
        #     drone_assigned_mask = (drone_for_request == d)  # [B, n_request]
        #     if drone_assigned_mask.any():
        #         cur_spaces_drone[:, d] -= (requests["volumn"] * drone_assigned_mask).sum(dim=-1)  # [B]

        #################### Decoding courier Actions. ####################
        # 获取当前处于不同阶段等待被搭载的请求
        stage1_request_mask = (cur_courier_request == 0).all(-2) #[B, R]
        stage2_request_mask = (cur_courier_request == 3).any(-2) & (cur_drone_request == 0).all(-2)
        stage3_request_mask = (cur_drone_request == 4).any(-2) & (cur_courier_request != 5).all(-2) & (cur_courier_request != 2).all(-2)
        
        all_couriers_decode_idx = torch.empty(batch_size, 0, dtype=torch.int64, device=device)
        cur_courier_need_decode = couriers["time_left"] == 0
        # 若没有任何courier被解码，使用仅包含request的mask作为默认值
        all_requests_courier_decode_mask = all_requests_decode_mask.clone()
        
        if not self.use_ar:
            cur_courier_request = _global["courier_request"].clone()

        for _idx in range(n_courier):
            courier_decode_idx = cur_courier_need_decode.to(torch.int64).argmax(-1)
            courier_decode_mask = cur_courier_need_decode[batch_arange, courier_decode_idx]
            if (~courier_decode_mask).all():
                break
            # 每轮被解码的车辆id
            if not self.use_ar:
                # conflict handler
                pickuping_mask = requests_couriers_action == courier_decode_idx[:, None]
                overvolumn_mask = ((pickuping_mask * requests["volumn"][batch_arange[:, None], requests_couriers_action.clamp_max(n_courier - 1)]).cumsum(-1) > couriers["space"][batch_arange, courier_decode_idx][:, None]) * pickuping_mask
                requests_couriers_action[overvolumn_mask] = n_courier

            all_couriers_decode_idx = torch.cat((all_couriers_decode_idx, courier_decode_idx[:, None]), dim=1)

            if self.use_unbind_decode:
                input_emb = couriers_rep[batch_arange, courier_decode_idx]
            else:
                input_emb = self.rep_action_proj_in(torch.cat((couriers_rep[batch_arange, courier_decode_idx], last_action_rep), dim=-1))    
            
            # 等待清除的旧设计
            dec_mask = F.pad(all_requests_decode_mask, (0, _idx+1), "constant", True)
            all_requests_courier_decode_mask = dec_mask.clone()
            # When use_ar=False, each decoder call is independent, so no_ar_mask should match current input length (1)
            if not self.use_ar:
                self.decoder.reset_kvcache()
                no_ar_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            else:
                no_ar_mask = dec_mask.clone()
                no_ar_mask[..., -1] = True

            pointer_hidden = self.decoder.forward(
                (input_emb + self.pos_embd(courier_decode_idx)).unsqueeze(1),
                dec_mask=None if self.use_ar else no_ar_mask,
                rel_mat=None, # DO NOT introduce rel_mat in decoder
                rep_enc=all_rep, 
                enc_mask=all_mask,
                enc_rel_mat=None,
                use_kvcache=self.use_ar,
                token_type=TOKEN_COURIER_NEXT
            ).squeeze(1)
            
            # pointer_hidden [B, n_embed]
            # drones_rep_with_nopickup [B, n_node, n_embed]
            # action_logits [B, n_node]
            
            action_logits = (pointer_hidden.unsqueeze(-2) @ nodes_rep.transpose(-1, -2)).squeeze(-2)

            # 试一下加temperature = 2 增强探索
            # self.temperature = 3
            # action_logits /= self.temperature
            action_probs = F.softmax(action_logits, -1) + 1e-5
            
            if self.only_heuristic:
                action_probs = torch.ones_like(action_probs)
            has_capacity = (couriers["space"][batch_arange, courier_decode_idx] != 0)[:, None]
            # 计算启发式概率
            node_have_request_from = (
                stage1_request_mask.unsqueeze(-1) &
                request_node_from # [B, R, N]
            ).any(-2) & has_capacity # [B, N]
            node_have_request_station2 = (
                stage3_request_mask.unsqueeze(-1) &
                request_node_station2 # [B, R, N]
            ).any(-2) & has_capacity # [B, N]
            
            node_have_request_to_pickup = node_have_request_from | node_have_request_station2
            
            node_have_request_to = (
                (cur_courier_request[batch_arange, courier_decode_idx] == 2).unsqueeze(-1) &
                (cur_drone_request == 4).any(-2).unsqueeze(-1) &
                request_node_to
            ).any(-2)

            # 计算去往潜在station的概率
            movement_costs = _global["node_node"][batch_arange, couriers["target"][batch_arange, courier_decode_idx]].to(torch.float32)  # [B, N]
            courier_has_stage1_requests = ((cur_courier_request[batch_arange, courier_decode_idx] == 2) & (~((cur_courier_request == 3).any(dim=1)))).any(-1) # [B, R]
            station_node_indices = _global["station_idx"][batch_arange, :-1] # [B, S]
            movement_costs_to_stations = movement_costs.gather(1, station_node_indices) # [B, S]
            
            avg_distance_station = dist_label[:, None] * 0.1
            # avg_distance_station = dist_label[:, None]  # [B, 1]
            station_distance_prob = (avg_distance_station / (movement_costs_to_stations + 1e-6)).clamp_max(1.) # [B, S]

            node_is_potential_station1 = torch.zeros(batch_size, n_node, dtype=torch.bool, device=device)
            # Mark stations as potential targets if courier has requests
            relevant_batches = courier_has_stage1_requests
            if relevant_batches.any():
                node_is_potential_station1[relevant_batches] = _global["station_mask"][relevant_batches]

            # 筛选top k个候选
            top_from, top_station2 = 3,3
            def get_topk_mask(full_mask, costs, k):
                """
                保留 full_mask 中 costs 最小的 k 个节点。
                返回: Top-K Mask (只包含最近的k个), Invalid Mask (包含剩余被剔除的请求节点)
                """
                # 这里的 mask 操作是为了只对有请求的节点进行排序，没请求的设为无穷大
                valid_costs = torch.where(full_mask, costs, torch.tensor(float('inf'), device=costs.device))
                # 防止 k 大于总节点数
                curr_k = min(k, full_mask.size(-1))
                # 找出距离最小的 k 个索引 (largest=False)
                # topk_vals: [B, k], topk_inds: [B, k]
                _, topk_inds = torch.topk(valid_costs, k=curr_k, dim=-1, largest=False)
                # 生成 Top-K 掩码
                topk_mask = torch.zeros_like(full_mask).scatter(-1, topk_inds, True)
                # 确保只保留原始 mask 中为 True 的位置 (处理 padding 或无效节点)
                final_topk_mask = topk_mask & full_mask
                # 找出那些原本有请求，但因为距离太远被剔除的节点 (用于后续 mask 掉概率)
                ignored_request_mask = full_mask & (~final_topk_mask)
                return final_topk_mask, ignored_request_mask
            
            topk_req_from, ignored_from = get_topk_mask(node_have_request_from, movement_costs, top_from)
            topk_req_station2, ignored_station2 = get_topk_mask(node_have_request_station2, movement_costs, top_station2)
            all_ignored_requests = ignored_from | ignored_station2

            additional_probs = torch.ones_like(movement_costs, dtype=torch.float32)
            additional_probs = torch.where(
                node_have_request_to_pickup | node_is_potential_station1 | node_have_request_to,
                0.1,
                self.other_node_prob
            )
            
            # Apply station distance heuristic
            if node_is_potential_station1.any():
                full_station_probs = torch.zeros_like(additional_probs)
                full_station_probs.scatter_(1, station_node_indices, station_distance_prob)
                # 取二者中更大的一个启发
                updated_station_probs = torch.max(full_station_probs, full_station_probs)
                additional_probs = torch.where(node_is_potential_station1, updated_station_probs, additional_probs)

            additional_probs[node_have_request_to] = 2
            additional_probs[all_ignored_requests] = 0
            action_probs = action_probs * additional_probs
            
            action_probs = action_probs / (action_probs.sum(-1, keepdim=True) + 1e-6)
            no_action_mask = (action_probs == .0).all(-1)
            action_probs[batch_arange[no_action_mask], couriers["target"][batch_arange, courier_decode_idx][no_action_mask]] = 1.0
           
            # 只要 logits 不是全 -inf 就可以
            cate = Categorical(probs=action_probs)
            if is_decoding:
                if deterministic:
                    action = cate.probs.argmax(dim=-1)
                else:
                    action = cate.sample()
                couriers_action[batch_arange[courier_decode_mask], courier_decode_idx[courier_decode_mask]] = action[courier_decode_mask]
            else:
                action = couriers_action[batch_arange, courier_decode_idx]

            action_log_prob = cate.log_prob(action)
            log_prob[courier_decode_mask] += action_log_prob[courier_decode_mask]
            entropy[courier_decode_mask] += cate.entropy()[courier_decode_mask]
            if self.use_ar:
                last_action_rep = nodes_rep[batch_arange, action]
            cur_courier_need_decode[batch_arange[courier_decode_mask], courier_decode_idx[courier_decode_mask]] = False

            if is_decoding:
                # 检查courier选择的action是否是station
                selected_actions = action[courier_decode_mask]  # [B_active]
                # B_active corresponds to batch_arange[courier_decode_mask]
                active_batch_idx = batch_arange[courier_decode_mask]
                active_courier_idx = courier_decode_idx[courier_decode_mask]
                is_station_mask = _global["station_mask"][active_batch_idx, selected_actions]  # [B_active]
                
                # 找出携带订单的courier（stage1阶段，cur_courier_request == 2）
                is_stage1_on_current = (cur_courier_request[active_batch_idx, active_courier_idx] == 2) & (~(cur_courier_request[active_batch_idx] == 3).any(dim=1))
                courier_has_stage1_requests_mask = is_stage1_on_current.any(-1)  # [B_active]
                
                # 需要更新station的courier：选择了station且携带订单
                # 这里动态指派被装载订单的station1和station2
                need_update_mask = is_station_mask & courier_has_stage1_requests_mask
                
                if need_update_mask.any():
                    update_batch_idx = active_batch_idx[need_update_mask]
                    update_courier_idx = active_courier_idx[need_update_mask]
                    update_station_nodes = selected_actions[need_update_mask]  # [B_update]
                    # 向量化更新
                    requests_to_update_mask = is_stage1_on_current[need_update_mask] # [B_update, R]
                    
                    r_indices = torch.arange(n_request, device=device).unsqueeze(0).expand(len(update_batch_idx), -1)
                    b_indices = update_batch_idx.unsqueeze(1).expand(-1, n_request)
                    
                    flat_b = b_indices[requests_to_update_mask]
                    flat_r = r_indices[requests_to_update_mask]
                    flat_s = update_station_nodes.unsqueeze(1).expand(-1, n_request)[requests_to_update_mask]
                    
                    station1_action[flat_b, flat_r] = flat_s
                    request_node_station1[flat_b, flat_r, flat_s] = True
                    
                    # Update station2 要和station1不同
                    req_to_nodes = requests["to"][flat_b, flat_r] # [N_updates]
                    batch_stations = _global["station_idx"][flat_b, :-1] # [N_updates, S]
                    dists_to_nodes = _global["node_node"][flat_b, req_to_nodes, :].to(torch.float32) # [N_updates, N]
                    dists_to_stations = dists_to_nodes.gather(1, batch_stations) # [N_updates, S]
                    exclude_mask = (batch_stations == flat_s.unsqueeze(1)) # [N_updates, S]
                    dists_to_stations.masked_fill_(exclude_mask, float('inf'))
                    
                    min_s_idx = dists_to_stations.argmin(dim=1) # [N_updates]
                    station2_nodes = batch_stations.gather(1, min_s_idx.unsqueeze(1)).squeeze(1) # [N_updates]
                    
                    station2_action[flat_b, flat_r] = station2_nodes
                    request_node_station2[flat_b, flat_r, station2_nodes] = True
        #################### Decoding Drone Actions. ####################
        all_drones_decode_idx = torch.empty(batch_size, 0, dtype=torch.int64, device=device)
        cur_drone_need_decode = drones["time_left"] == 0
        
        # 无人机接单过的直接前往目的地
        has_request_stage2 = (cur_drone_request == 2).any(dim=-1) # [B, D]
        special_drone_idx = (cur_drone_need_decode & has_request_stage2).nonzero(as_tuple=True)
        if len(special_drone_idx[0]) > 0:
            request_idx = (cur_drone_request[special_drone_idx] == 2).int().argmax(dim=-1)
            drones_action[special_drone_idx] = requests["station2"][special_drone_idx[0], request_idx]
            cur_drone_need_decode[special_drone_idx] = False
        
        # TODO 有bug
        # dec_mask 可能未定义（例如没有courier可解码），使用上面的默认mask
        if not self.use_ar:
            cur_drone_request = _global["drone_request"].clone()

        for _idx in range(n_drone):
            drone_decode_idx = cur_drone_need_decode.to(torch.int64).argmax(-1)
            drone_decode_mask = cur_drone_need_decode[batch_arange, drone_decode_idx]
            if (~drone_decode_mask).all():
                break
            all_drones_decode_idx = torch.cat((all_drones_decode_idx, drone_decode_idx[:, None]), dim=1)

            if self.use_unbind_decode:
                input_emb = drones_rep[batch_arange, drone_decode_idx]
            else:
                input_emb = self.rep_action_proj_in(torch.cat((drones_rep[batch_arange, drone_decode_idx], last_action_rep), dim=-1))

            dec_mask = F.pad(all_requests_courier_decode_mask, (0, _idx+1), "constant", True)
            # When use_ar=False, each decoder call is independent, so no_ar_mask should match current input length (1)
            if not self.use_ar:
                self.decoder.reset_kvcache()
                no_ar_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            else:
                no_ar_mask = dec_mask.clone()
                no_ar_mask[..., -1] = True
            
            pointer_hidden = self.decoder.forward(
                (input_emb + self.pos_embd(drone_decode_idx)).unsqueeze(1),
                dec_mask=None if self.use_ar else no_ar_mask,
                rel_mat=None, # DO NOT introduce rel_mat in decoder
                rep_enc=all_rep, 
                enc_mask=all_mask,
                enc_rel_mat=None,
                use_kvcache=self.use_ar,
                token_type=TOKEN_DRONE_NEXT
            ).squeeze(1)
            # pointer_hidden [B, n_embed]
            # drones_rep_with_nopickup [B, n_node, n_embed]
            # action_logits [B, n_node]
            action_logits = (pointer_hidden.unsqueeze(-2) @ stations_rep.transpose(-1, -2)).squeeze(-2)
            action_probs = F.softmax(action_logits, -1) + 1e-5
            if self.only_heuristic:
                action_probs = torch.ones_like(action_probs)
            
            node_have_request_station1 = (
                stage2_request_mask.unsqueeze(-1) &
                request_node_station1  # [B, R, N]
            ).any(-2)  # [B, N]
            
            # 计算启发式概率
            current_positions = drones["target"][batch_arange, drone_decode_idx]
            station_indices = torch.argmax((_global["station_idx"][batch_arange] == current_positions.unsqueeze(1)).float(), dim=1)  # [B]
            
            movement_costs = _global["station_node_node"][batch_arange, station_indices]  # [B, S]
            avg_distance = dist_label_drone[:, None]  # [B, 1]
            distance_prob = (avg_distance / (movement_costs + 1e-6)).clamp_max(1.) * 0.1
            assert distance_prob.isfinite().all()
            
            has_capacity = (drones["space"][batch_arange, drone_decode_idx] != 0)[:, None]
            station_request_mask = node_have_request_station1.gather(1, _global["station_idx"][..., :-1])  # [B, S]
        
            additional_probs = torch.where(
                has_capacity & station_request_mask,
                distance_prob,
                0
            )
            action_probs = action_probs * additional_probs
            no_action_mask = (action_probs == 0.0).all(-1)

            target_nodes = drones["target"][batch_arange, drone_decode_idx][no_action_mask]
            station_indices = torch.argmax((_global["station_idx"][batch_arange[no_action_mask]] == target_nodes.unsqueeze(1)).float(), dim=1)
        
            action_probs[batch_arange[no_action_mask], station_indices] = 1.0
            
            cate = Categorical(probs=action_probs)
            if is_decoding:
                if deterministic:
                    action = cate.probs.argmax(dim=-1)
                else:
                    action = cate.sample()
                node_choice = _global["station_idx"][batch_arange, action]
                drones_action[batch_arange[drone_decode_mask], drone_decode_idx[drone_decode_mask]] = node_choice[drone_decode_mask]
            else:
                raw_action = drones_action[batch_arange, drone_decode_idx]
                action = (_global["station_idx"] == raw_action.unsqueeze(-1)).int().argmax(dim=-1)
            # TODO: masked_add
            action_log_prob = cate.log_prob(action)
            log_prob[drone_decode_mask] += action_log_prob[drone_decode_mask]
            entropy[drone_decode_mask] += cate.entropy()[drone_decode_mask]
            if self.use_ar:
                last_action_rep = nodes_rep[batch_arange, action]
            cur_drone_need_decode[batch_arange[drone_decode_mask], drone_decode_idx[drone_decode_mask]] = False
        self.decoder.reset_kvcache()
        
        return {"station1": station1_action, 
                "station2": station2_action, 
                "courier": couriers_action, 
                "drone": drones_action, 
                "request_courier": requests_couriers_action, 
                "request_drone": requests_drones_action}, log_prob, entropy, values