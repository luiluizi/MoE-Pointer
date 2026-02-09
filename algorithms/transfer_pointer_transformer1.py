from itertools import product, permutations
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions import Categorical

from .myModels import BatchNorm, Encoder, Decoder, MLP, assign_symmetric, IterMixin, MulConstant, Identity, NoEmbedding, PointerWithPrior


class MultiAgentPointerTransformer(nn.Module, IterMixin):

    def __init__(self, n_enc_bloc, n_dec_block, n_embd, n_head, rel_dim, device, max_len=1100, env_args=None, use_ar=True, use_heur_req=True, use_heur_veh=True, use_relation=True, use_tsp=True, use_node_emb=False, use_unbind_decode=False, use_nearest_station = False, only_heuristic=False, hypers=None):
        super().__init__()
        IterMixin.__init__(self)
        self.cnt = 0
        print("raw!!!!!!!!!!!!!!!!!!!!!!!!!")
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
        self.use_tsp = use_tsp
        self.use_node_emb = use_node_emb
        self.use_unbind_decode = use_unbind_decode
        self.use_nearest_station = use_nearest_station
        
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

        if self.use_ar:
            self.pos_embd = nn.Embedding(max_len, n_embd)
        else:
            self.pos_embd = NoEmbedding(n_embd)

        self.node_node_proj_in = MLP(1, n_embd, rel_dim, scale=4., enable_scale=True)
        self.vehicle_node_embd = nn.Sequential(nn.Embedding(2, n_embd), MLP(n_embd, rel_dim, pre_act=True, scale=12., enable_scale=True))
        self.request_node_embd = nn.Sequential(nn.Embedding(3, n_embd), MLP(n_embd, rel_dim, pre_act=True, scale=15., enable_scale=True))
        self.vehicle_request_embd = nn.Sequential(nn.Embedding(4, n_embd), MLP(n_embd, rel_dim, pre_act=True, scale=8., enable_scale=True))
        self.request_request_dist_proj_in = MLP(4, n_embd, rel_dim, scale=8., enable_scale=True)
        
        self.rep_action_proj_in = nn.Linear(n_embd * 2, n_embd)
        self.value_linear = nn.Linear(n_embd, 1)
        # self.value_linear = nn.Sequential(
        #     nn.Linear(n_embd, n_embd),
        #     nn.GELU(),
        #     nn.Linear(n_embd, 1)
        # )
        
        self.courier_pointer = PointerWithPrior(n_embd)
        
        self.SOS = nn.Parameter(torch.randn(n_embd)) # should be init to correct scale?
        self.NOACTION = nn.Parameter(torch.randn(n_embd))
        self.node_emb = nn.Parameter(torch.randn(500, n_embd))
        if self.use_node_emb:
            self.register_buffer("stored_use_node_emb", torch.tensor(self.use_node_emb))

        if not self.only_heuristic:
            self.encoder = Encoder(n_enc_bloc, n_embd, n_head)
            self.decoder = Decoder(n_dec_block, n_embd, n_head)
            # self.c_head = nn.Linear(n_embd, n_embd)
            # self.d_head = nn.Linear(n_embd, n_embd)
            # self.r_head = nn.Linear(n_embd, n_embd)
            self.c_head = Identity()
            self.d_head = Identity()
            self.r_head = Identity()
        else:
            self.encoder = Identity()
            self.decoder = Identity()
            self.c_head = Identity()
            self.d_head = Identity()
            self.r_head = Identity()
            # self.decoder1 = Identity()

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
        
        # 临时加入，用于缓存station_assgin
        self.station1_action_temp = None
        self.station2_action_temp = None
        

    def forward(self, obs, input_actions=None, deterministic=False, only_critic=False, heuristic_weight = 1.0):  
        self.cnt += 1
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
            # _global["n_stage3"].to(torch.float),
            _global["n_exist_requests"].to(torch.float),
            # _global["n_stage1"].to(torch.float),
            # _global["n_stage2"].to(torch.float),
            # _global["n_stage3"].to(torch.float),
        ), dim=1)))
        
        
        # 放在 forward 开头、在计算 lens 之前或之后立刻打印
        batch_size = _global["n_exist_requests"].shape[0]
        device = nodes["n_courier"].device

        n_node = nodes["n_courier"].shape[1]
        n_courier = couriers["capacity"].shape[1]
        n_drone = drones["target"].shape[1]
        n_request = requests["value"].shape[1]
        
        lens = [n_node, n_courier, n_drone, n_request, 1]
        acc_lens = np.cumsum(lens)
        # mask掉未consider的request
        all_mask = torch.ones(batch_size, sum(lens), dtype=torch.bool, device=device)
        all_mask[:, sum(lens[:3]):sum(lens[:4])][torch.arange(lens[3], device=device) >= _global["n_consider_requests"][:, None]] = False
        requests_mask = torch.arange(lens[3], device=device) < _global["n_consider_requests"][:, None]
        batch_arange = torch.arange(batch_size, device=device)
        courier_arange = torch.arange(n_courier, device=device)
        drone_arrange = torch.arange(n_drone, device=device)
        request_arange = torch.arange(n_request, device=device)

        all_rep = self.encoder.forward(torch.cat((
            nodes_feature, 
            couriers_feature, 
            drones_feature,
            requests_feature, 
            global_feature.unsqueeze(-2)
        ), dim=-2) + self.pos_embd(torch.arange(acc_lens[-1], device=device)[None]), all_mask)
        
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

        all_station_decode_idx = torch.empty(batch_size, 0, dtype=torch.int64, device=device)
        all_station_decode_mask = torch.empty(batch_size, 0, dtype=torch.bool, device=device)
        
        # TODO 11.7 
        # 两种实现方式，一个是分别决策station1，station2，在两个循环中完成
        # 另一个是在一个循环里面做决策，就像request决策stage1，stage2一样
        # 我认为不需要上来对于所有没决策的订单全都决策出来，吃显存厉害
        # 暂时没想法，先直接决策完吧
        
        # 实现一
        if not self.use_nearest_station:
            for _idx in range(n_request):
                # 检测处于stage1的request
                stage1_mask = (cur_courier_request == 0).all(-2) & (requests["station1"] == n_node)
                # [B]保存第一个
                requests_decode_idx = stage1_mask.to(torch.int64).argmax(-1)
                requests_decode_mask = stage1_mask[batch_arange, requests_decode_idx]
                if (~requests_decode_mask).all():
                    break
                all_station_decode_idx = torch.cat((all_station_decode_idx, requests_decode_idx[:, None]), dim=1) # [B, n_decoded_requests]
                all_station_decode_mask = torch.cat((all_station_decode_mask, requests_decode_mask[:, None]), dim=1) # [B, n_decoded_requests]

                # 是否use_unbind_decode
                if self.use_unbind_decode:
                    input_emb = requests_rep[batch_arange, requests_decode_idx]
                else:
                    # TODO proj是否可以拆分
                    input_emb = self.rep_action_proj_in(torch.cat((requests_rep[batch_arange, requests_decode_idx], last_action_rep), dim=-1))

                dec_mask = F.pad(all_station_decode_mask, (0, _idx+1), "constant", True)
                no_ar_mask = dec_mask.clone()
                no_ar_mask[..., -1] = True

                pointer_hidden = self.decoder.forward(
                    (input_emb + self.pos_embd(requests_decode_idx)).unsqueeze(1), 
                    None if self.use_ar else no_ar_mask,
                    all_rep, 
                    all_mask, 
                    use_kvcache=True
                ).squeeze(1)

                action_logits = (pointer_hidden.unsqueeze(-2) @ stations_rep.transpose(-1, -2)).squeeze(-2)
                action_probs = F.softmax(action_logits, -1) + 1e-5
                
                if self.only_heuristic:
                    action_probs = torch.ones_like(action_probs)
                
                # 只保留最近的两个station
                current_positions = requests["from"][batch_arange, requests_decode_idx]
                movement_costs = _global["node_node"][batch_arange, current_positions]
                station_costs = torch.gather(movement_costs, 1, _global["station_idx"][...,:-1])
                sorted_vals, sorted_idx = torch.sort(station_costs, dim=1)
                top2_idx = sorted_idx[:, :self.top] # [B, 2]
                top2_idx = top2_idx[:,1].unsqueeze(1)
                
                mask = torch.zeros_like(station_costs, dtype=torch.bool)  # [B, n_station]
                mask.scatter_(1, top2_idx, True)  # 将 top2 的位置标为 True
                
                # avg_distance = dist_label_drone[:, None]  # [B, 1]
                # distance_prob = (avg_distance / (station_costs + 1e-6)).clamp_max(1.)
                
                action_probs = action_probs * mask
                # import pdb
                # pdb.set_trace()
                
                cate = Categorical(probs=action_probs)
                if is_decoding:
                    if deterministic:
                        action = cate.probs.argmax(dim=-1)
                    else:
                        action = cate.sample()
                    node_choice = _global["station_idx"][batch_arange, action]
                    # 确保全部已经完成指派
                    assert(node_choice != n_node).any()
                    station1_action[batch_arange[requests_decode_mask], requests_decode_idx[requests_decode_mask]] = node_choice[requests_decode_mask]
                else:
                    raw_action = station1_action[batch_arange, requests_decode_idx]
                    action = (_global["station_idx"] == raw_action.unsqueeze(-1)).int().argmax(dim=-1)
                # 完成指派
                # TODO 是否有必要clone 到时候训练直接试吧      
                requests["station1"][batch_arange[requests_decode_mask], requests_decode_idx[requests_decode_mask]] = station1_action[batch_arange[requests_decode_mask], requests_decode_idx[requests_decode_mask]] 
                action_log_prob = cate.log_prob(action)
                log_prob[requests_decode_mask] += action_log_prob[requests_decode_mask]
                entropy[requests_decode_mask] += cate.entropy()[requests_decode_mask]
                last_action_rep = nodes_rep[batch_arange, action]

            # 决策station2
            for _idx in range(n_request):
                # 检测处于stage2的request
                stage2_mask = (cur_courier_request == 3).any(-2) & (cur_drone_request == 0).all(-2) & (requests["station2"] == n_node) & (requests["station1"] != n_node)
                # [B]保存第一个
                requests_decode_idx = stage2_mask.to(torch.int64).argmax(-1)
                requests_decode_mask = stage2_mask[batch_arange, requests_decode_idx]
                if (~requests_decode_mask).all():
                    break
                all_station_decode_idx = torch.cat((all_station_decode_idx, requests_decode_idx[:, None]), dim=1) # [B, n_decoded_requests]
                all_station_decode_mask = torch.cat((all_station_decode_mask, requests_decode_mask[:, None]), dim=1) # [B, n_decoded_requests]

                # 是否use_unbind_decode
                if self.use_unbind_decode:
                    input_emb = requests_rep[batch_arange, requests_decode_idx]
                else:
                    # TODO proj是否可以拆分
                    input_emb = self.rep_action_proj_in(torch.cat((requests_rep[batch_arange, requests_decode_idx], last_action_rep), dim=-1))

                dec_mask = F.pad(all_station_decode_mask, (0, _idx+1), "constant", True)
                no_ar_mask = dec_mask.clone()
                no_ar_mask[..., -1] = True

                pointer_hidden = self.decoder.forward(
                    (input_emb + self.pos_embd(requests_decode_idx)).unsqueeze(1), 
                    None if self.use_ar else no_ar_mask,
                    all_rep, 
                    all_mask, 
                    use_kvcache=True
                ).squeeze(1)

                action_logits = (pointer_hidden.unsqueeze(-2) @ stations_rep.transpose(-1, -2)).squeeze(-2)
                action_probs = F.softmax(action_logits, -1) + 1e-5            
                
                if self.only_heuristic:
                    action_probs = torch.ones_like(action_probs)
                # 不能选同一个station  
                # station1的node id转换为station id
                station_id_mask = _global["station_idx"] == requests["station1"][batch_arange, requests_decode_idx].unsqueeze(1)
                action_probs[batch_arange, torch.argmax(station_id_mask.int(), dim=1)] = 0.0         
                
                current_positions = requests["to"][batch_arange, requests_decode_idx]
                movement_costs = _global["node_node"][batch_arange, current_positions]
                station_costs = torch.gather(movement_costs, 1, _global["station_idx"][...,:-1])
                sorted_vals, sorted_idx = torch.sort(station_costs, dim=1)
                top2_idx = sorted_idx[:, :self.top]  # [B, 2]
                mask = torch.zeros_like(station_costs, dtype=torch.bool)  # [B, n_station]
                mask.scatter_(1, top2_idx, True)  # 将 top2 的位置标为 True
                                
                # avg_distance = dist_label_drone[:, None]  # [B, 1]
                # distance_prob = (avg_distance / (station_costs + 1e-6)).clamp_max(1.)
                
                action_probs = action_probs * mask
                
                
                cate = Categorical(probs=action_probs)
                if is_decoding:
                    if deterministic:
                        action = cate.probs.argmax(dim=-1)
                    else:
                        action = cate.sample()
                    node_choice = _global["station_idx"][batch_arange, action]
                    # 确保全部已经完成指派
                    assert(node_choice != n_node).any()
                    station2_action[batch_arange[requests_decode_mask], requests_decode_idx[requests_decode_mask]] = node_choice[requests_decode_mask]
                else:
                    raw_action = station2_action[batch_arange, requests_decode_idx]
                    action = (_global["station_idx"] == raw_action.unsqueeze(-1)).int().argmax(dim=-1)
                # 完成指派
                # TODO 是否有必要clone
                requests["station2"][batch_arange[requests_decode_mask], requests_decode_idx[requests_decode_mask]] = station2_action[batch_arange[requests_decode_mask], requests_decode_idx[requests_decode_mask]]            
                action_log_prob = cate.log_prob(action)
                log_prob[requests_decode_mask] += action_log_prob[requests_decode_mask]
                entropy[requests_decode_mask] += cate.entropy()[requests_decode_mask]
                last_action_rep = nodes_rep[batch_arange, action]
        
        # 暂时，指派订单的station1和station2均是距离from或to最近的station
        # 在eval时候需要重算一遍

        if self.use_nearest_station:
            dist_matrix = _global["node_node"].to(torch.float32)
            for b in range(batch_size):
                stations = _global["station_idx"][b,:-1]
                request_from = requests["from"][b]
                request_to = requests["to"][b]
                
                from_dists = dist_matrix[b, request_from[:, None], stations[None, :]]
                min_station1_idx = from_dists.argmin(dim=1)
                station1_action[b] = stations[min_station1_idx]
                
                to_dists = dist_matrix[b, request_to[:, None], stations[None, :]]
                
                # 创建副本并排除station1对应的站点
                to_dists_excluded = to_dists.clone()
                # 构建掩码：标记每个请求的station1位置
                exclude_mask = torch.zeros_like(to_dists_excluded, dtype=torch.bool)
                exclude_mask[torch.arange(len(min_station1_idx)), min_station1_idx] = True
                # 将被排除的站点距离设为无穷大
                to_dists_excluded = to_dists_excluded.masked_fill(exclude_mask, float('inf'))
                
                # 在排除station1的站点中找最近的作为station2
                min_station2_idx = to_dists_excluded.argmin(dim=1)
                station2_action[b] = stations[min_station2_idx]
                
                # 验证所有请求的station1和station2均不同
                assert (station1_action[b] != station2_action[b]).all(), "Station assignment conflict not resolved"
            self.station1_action_temp = station1_action.clone()
            self.station2_action_temp = station2_action.clone()
            assert(self.station1_action_temp!=self.station2_action_temp).all()   
            requests["station1"] = self.station1_action_temp
            requests["station2"] = self.station2_action_temp
            station1_action = self.station1_action_temp
            station2_action = self.station2_action_temp
            
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
        
        # TODO 请求的问题，要解码的请求太多了 每多解码一次加280m
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
            
            no_ar_mask = torch.zeros(batch_size, _idx+1, dtype=torch.bool, device=device)
            no_ar_mask[..., -1] = True; # we use no_ar_mask to mask previous tokens, which equivalent to input separatly. 
            
            pointer_hidden = self.decoder.forward(
                (input_emb + self.pos_embd(requests_decode_idx)).unsqueeze(1), 
                None if self.use_ar else no_ar_mask,
                all_rep, 
                all_mask, 
                use_kvcache=True
            ).squeeze(1)
            
            pointer_hidden = self.r_head(pointer_hidden)

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
        
        for _idx in range(n_courier):
            courier_decode_idx = cur_courier_need_decode.to(torch.int64).argmax(-1)
            courier_decode_mask = cur_courier_need_decode[batch_arange, courier_decode_idx]
            if (~courier_decode_mask).all():
                break
            # 每轮被解码的车辆id
            all_couriers_decode_idx = torch.cat((all_couriers_decode_idx, courier_decode_idx[:, None]), dim=1)

            if self.use_unbind_decode:
                input_emb = couriers_rep[batch_arange, courier_decode_idx]
            else:
                input_emb = self.rep_action_proj_in(torch.cat((couriers_rep[batch_arange, courier_decode_idx], last_action_rep), dim=-1))    
            
            # 等待清除的旧设计
            dec_mask = F.pad(all_requests_decode_mask, (0, _idx+1), "constant", True)
            no_ar_mask = dec_mask.clone()
            no_ar_mask[..., -1] = True
            
            pointer_hidden = self.decoder.forward(
                (input_emb + self.pos_embd(courier_decode_idx)).unsqueeze(1),
                None if self.use_ar else no_ar_mask,
                all_rep, 
                all_mask, 
                use_kvcache=True
            ).squeeze(1)
            
            # aware att
            pointer_hidden = self.c_head(pointer_hidden)
            # pointer_hidden [B, n_embed]
            # drones_rep_with_nopickup [B, n_node, n_embed]
            # action_logits [B, n_node]
            
            
            action_logits = (pointer_hidden.unsqueeze(-2) @ nodes_rep.transpose(-1, -2)).squeeze(-2)
            
            # import pdb
            # pdb.set_trace()
            
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
            
            node_have_request_station1 = (
                (cur_courier_request[batch_arange, courier_decode_idx] == 2).unsqueeze(-1) &
                (cur_drone_request != 4).all(-2).unsqueeze(-1) &
                request_node_station1
            ).any(-2)
            
            node_have_request_to = (
                (cur_courier_request[batch_arange, courier_decode_idx] == 2).unsqueeze(-1) &
                (cur_drone_request == 4).any(-2).unsqueeze(-1) &
                request_node_to
            ).any(-2)

            current_positions = couriers["target"][batch_arange, courier_decode_idx]
            movement_costs = _global["node_node"][batch_arange, current_positions]  # [B, N]
            avg_distance = dist_label[:, None] * 0.4 # [B, 1]  调小可以提高sample的准确率
            pri_index = 0.25 * (self.env_args["max_capacity"] / (n_node - n_station))
            distance_prob = (avg_distance / (movement_costs + 1e-6)).clamp_max(1.) * 0.1
            # 筛选top k个候选
            a1, a2, a3, a4 = 3,3,1,1
            
            import pdb
            # pdb.set_trace()
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
            
            topk_req_from, ignored_from = get_topk_mask(node_have_request_from, movement_costs, a1)
            topk_req_station2, ignored_station2 = get_topk_mask(node_have_request_station2, movement_costs, a2)
            # topk_req_station1, ignored_station1 = get_topk_mask(node_have_request_station1, movement_costs, a3)
            # topk_req_to, ignored_to = get_topk_mask(node_have_request_to, movement_costs, a4)
            all_ignored_requests = ignored_from | ignored_station2
            import pdb
            # pdb.set_trace()
            
            additional_probs = torch.ones_like(distance_prob)
            
            additional_probs = torch.where(
                node_have_request_to_pickup | node_have_request_station1 | node_have_request_to,
                0.1,
                self.other_node_prob
            )
            additional_probs[node_have_request_station1] = 1
            additional_probs[node_have_request_to] = 1
            additional_probs[all_ignored_requests] = 0
            action_probs = action_probs * additional_probs
            
            action_probs = action_probs / (action_probs.sum(-1, keepdim=True) + 1e-6)
            no_action_mask = (action_probs == .0).all(-1)
            action_probs[batch_arange[no_action_mask], couriers["target"][batch_arange, courier_decode_idx][no_action_mask]] = 1.0
            
            # 留在原地，其他概率太小，并且有空间
            # 如果订单密集没必要
            # action_probs[batch_arange, couriers["target"][batch_arange, courier_decode_idx]] = 0.2
           
            # 只要 logits 不是全 -inf 就可以
            cate = Categorical(probs=action_probs)
            if is_decoding:
                if deterministic:
                    action = cate.probs.argmax(dim=-1)
                    # action = cate.sample()
                else:
                    action = cate.sample()
                couriers_action[batch_arange[courier_decode_mask], courier_decode_idx[courier_decode_mask]] = action[courier_decode_mask]
            else:
                action = couriers_action[batch_arange, courier_decode_idx]
            # TODO: masked_add
            action_log_prob = cate.log_prob(action)
            log_prob[courier_decode_mask] += action_log_prob[courier_decode_mask]
            entropy[courier_decode_mask] += cate.entropy()[courier_decode_mask]
            last_action_rep = nodes_rep[batch_arange, action]
            cur_courier_need_decode[batch_arange[courier_decode_mask], courier_decode_idx[courier_decode_mask]] = False
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

            dec_mask = F.pad(all_requests_decode_mask, (0, _idx+1), "constant", True)
            no_ar_mask = dec_mask.clone()
            no_ar_mask[..., -1] = True
            
            pointer_hidden = self.decoder.forward(
                (input_emb + self.pos_embd(drone_decode_idx)).unsqueeze(1),
                None if self.use_ar else no_ar_mask,
                all_rep, 
                all_mask, 
                use_kvcache=True
            ).squeeze(1)
            pointer_hidden = self.d_head(pointer_hidden)
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
            # action_probs = action_probs / (action_probs.sum(-1, keepdim=True) + 1e-6)
            
            no_action_mask = (action_probs == 0.0).all(-1)

            target_nodes = drones["target"][batch_arange, drone_decode_idx][no_action_mask]
            station_indices = torch.argmax((_global["station_idx"][batch_arange[no_action_mask]] == target_nodes.unsqueeze(1)).float(), dim=1)
        
            action_probs[batch_arange[no_action_mask], station_indices] = 1.0
            
            # target_nodes = drones["target"][batch_arange, drone_decode_idx]
            # station_indices = torch.argmax((_global["station_idx"][batch_arange] == target_nodes.unsqueeze(1)).float(), dim=1)
            # action_probs[batch_arange, station_indices] = 0.5
            
            # distance_prob = (dist_label[:, None] /_global["node_node"][batch_arange, drones["target"][batch_arange, drone_decode_idx]]).clamp_max(1.) * 0.1
            # assert distance_prob.isfinite().all()
            # additional_probs = torch.where(node_have_request_station1, distance_prob, self.other_node_prob)
            # additional_probs = additional_probs.gather(dim=1, index=_global["station_idx"][:, :-1])
            # action_probs = action_probs * additional_probs
            # 没有合法动作就待在原地
            # no_action_mask = (action_probs < 0.3).all(-1)
            # # 这里需要把target从node_id映射回station_id
            # target_nodes = drones["target"][batch_arange, drone_decode_idx][no_action_mask]
            # station_indices = torch.argmax((_global["station_idx"][batch_arange[no_action_mask]] == target_nodes.unsqueeze(1)).float(), dim=1)
   
            # action_probs[batch_arange[no_action_mask], station_indices] = 1
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
            last_action_rep = nodes_rep[batch_arange, action]
            cur_drone_need_decode[batch_arange[drone_decode_mask], drone_decode_idx[drone_decode_mask]] = False
        self.decoder.reset_kvcache()
        # self.decoder1.reset_kvcache()
        # import pdb
        # pdb.set_trace()
        
        return {"station1": station1_action, 
                "station2": station2_action, 
                "courier": couriers_action, 
                "drone": drones_action, 
                "request_courier": requests_couriers_action, 
                "request_drone": requests_drones_action}, log_prob, entropy, values