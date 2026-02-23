import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from .component.models import BatchNorm, Encoder, MLP, Attention, IterMixin
from envs.env import obs_func_map, trans_func_map, DroneTransferEnv


class MAPDP(nn.Module, IterMixin):

    @staticmethod
    def obs_func(env:DroneTransferEnv, obs, unobs):
        batch_size = env.batch_size
        n_request = obs["requests"]["from"].shape[1]
        device = env.device
        
        # Heuristic station selection
        dist_matrix = env._global["node_node"].to(torch.float32)
        station1_action = torch.full([batch_size, n_request], env.n_node, dtype=torch.int64, device=device)
        station2_action = torch.full([batch_size, n_request], env.n_node, dtype=torch.int64, device=device)
        
        for b in range(batch_size):
            stations = env._global["station_idx"][b, :-1]  # Exclude padding values
            request_from = obs["requests"]["from"][b]
            request_to = obs["requests"]["to"][b]
            # Calculate distances from request origins to all stations, select the nearest as station1
            from_dists = dist_matrix[b, request_from[:, None], stations[None, :]]
            min_station1_idx = from_dists.argmin(dim=1)
            station1_action[b] = stations[min_station1_idx]
            # Calculate distances from request destinations to all stations, exclude station1 and select the nearest as station2
            to_dists = dist_matrix[b, request_to[:, None], stations[None, :]]
            to_dists_excluded = to_dists.clone()
            exclude_mask = torch.zeros_like(to_dists_excluded, dtype=torch.bool)
            exclude_mask[torch.arange(len(min_station1_idx), device=device), min_station1_idx] = True
            to_dists_excluded = to_dists_excluded.masked_fill(exclude_mask, float('inf'))
            min_station2_idx = to_dists_excluded.argmin(dim=1)
            station2_action[b] = stations[min_station2_idx]
            # Verify station1 and station2 are different
            assert (station1_action[b] != station2_action[b]).all(), "Station assignment conflict not resolved"
        
        # Write station selection results to observation
        obs["requests"]["station1"] = station1_action.clone()
        obs["requests"]["station2"] = station2_action.clone()
        
        # Initialize start_node and task_target for couriers and drones
        if env._global["frame"] == 0:
            env.couriers["start_node"] = env.couriers["target"].clone()
            env.couriers["task_target"] = torch.arange(env.n_courier, device=device).repeat(batch_size, 1)
            env.drones["start_node"] = env.drones["target"].clone()
            # Task_target for drones needs to add offset of courier segment
            n_tot_requests = env.n_tot_requests
            env.drones["task_target"] = torch.arange(
                env.n_courier + 4 * n_tot_requests,
                env.n_courier + 4 * n_tot_requests + env.n_drone,
                device=device
            ).unsqueeze(0).repeat(batch_size, 1)
        # Add initialization values to observation
        obs["couriers"]["task_target"] = env.couriers["task_target"].clone()
        obs["couriers"]["start_node"] = env.couriers["start_node"].clone()
        obs["drones"]["task_target"] = env.drones["task_target"].clone()
        obs["drones"]["start_node"] = env.drones["start_node"].clone()

    @staticmethod
    def trans_func(env:DroneTransferEnv, actions):
        new_task_target_courier = actions["new_task_target_courier"]
        new_task_target_drone = actions["new_task_target_drone"]
    
        # Validate courier task_target range: [0, n_courier + 4*n_tot_requests)
        courier_max = env.n_courier + 4 * env.n_tot_requests
        assert ((new_task_target_courier >= 0) & (new_task_target_courier <= courier_max)).all()
        new_task_mask_courier = new_task_target_courier != courier_max
        env.couriers["task_target"][new_task_mask_courier] = new_task_target_courier[new_task_mask_courier]
        
        # Validate drone task_target range: [n_courier + 4*n_tot_requests, n_courier + 4*n_tot_requests + n_drone + 2*n_tot_requests)
        drone_min = env.n_courier + 4 * env.n_tot_requests+1
        drone_max = env.n_courier + 4 * env.n_tot_requests + env.n_drone + 2 * env.n_tot_requests
        assert ((new_task_target_drone >= drone_min) & (new_task_target_drone <= drone_max)).all()
        new_task_mask_drone = new_task_target_drone != drone_max
        env.drones["task_target"][new_task_mask_drone] = new_task_target_drone[new_task_mask_drone]

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

        # Environment parameters
        self.n_courier = env_args["n_courier"]
        self.n_drone = env_args["n_drone"]
        self.n_station = env_args["n_station"]
        self.n_tot_requests = env_args["n_init_requests"] + env_args["n_norm_requests"]

        # Shared parameters (for unified encoding)
        self.global_bn = BatchNorm(2)
        self.global_proj_in = nn.Linear(2, n_embd)
        self.request_proj_in = nn.Linear(4, n_embd)
        self.request_request_dist_proj_in = MLP(1, n_embd, rel_dim)
        self.coord_proj_in = nn.Linear(2, n_embd)
        self.encoder = Encoder(n_enc_bloc, n_embd, n_head, qkfeat_dim=rel_dim)

        # Shared value head
        self.value_linear = MLP(n_embd, n_embd, n_embd, 1)

        # Courier node type embedding parameters
        self.COURIER_DEPOT = nn.Parameter(torch.randn(n_embd))
        self.COURIER_STAGE1_PICKUP = nn.Parameter(torch.randn(n_embd))
        self.COURIER_STAGE3_PICKUP = nn.Parameter(torch.randn(n_embd))
        self.COURIER_STAGE1_DELIVERY = nn.Parameter(torch.randn(n_embd))
        self.COURIER_STAGE3_DELIVERY = nn.Parameter(torch.randn(n_embd))
        self.courier_depot_proj_in = nn.Linear(n_embd, n_embd)
        self.courier_stage1_pickup_proj_in = nn.Linear(n_embd, n_embd)
        self.courier_stage3_pickup_proj_in = nn.Linear(n_embd, n_embd)
        self.courier_stage1_delivery_proj_in = nn.Linear(n_embd, n_embd)
        self.courier_stage3_delivery_proj_in = nn.Linear(n_embd, n_embd)

        # Drone node type embedding parameters
        self.DRONE_DEPOT = nn.Parameter(torch.randn(n_embd))
        self.DRONE_STAGE2_PICKUP = nn.Parameter(torch.randn(n_embd))
        self.DRONE_STAGE2_DELIVERY = nn.Parameter(torch.randn(n_embd))
        self.drone_depot_proj_in = nn.Linear(n_embd, n_embd)
        self.drone_stage2_pickup_proj_in = nn.Linear(n_embd, n_embd)
        self.drone_stage2_delivery_proj_in = nn.Linear(n_embd, n_embd)

        self.courier_proj_h = nn.Linear(env_args["n_courier"]*(1+n_embd)+2*n_embd+1, n_embd)
        self.drone_proj_h = nn.Linear(env_args["n_drone"]*(1+n_embd)+2*n_embd+1, n_embd)
        self.cross_attn = Attention(n_embd, n_head)
        self.to_q_g = nn.Linear(n_embd, n_embd)
        self.to_k_g = nn.Linear(n_embd, n_embd)
        self.NO_TARGET = nn.Parameter(torch.randn(n_embd))
        self.NO_ACTION = nn.Parameter(torch.randn(n_embd))


    def forward(self, obs, input_actions=None, deterministic=False, only_critic=False):
        nodes = obs["nodes"]
        couriers = obs["couriers"]
        drones = obs["drones"]
        requests = obs["requests"]
        _global = obs["global"]

        batch_size = _global["n_exist_requests"].shape[0]
        n_courier = self.n_courier
        n_drone = self.n_drone
        n_request = requests["value"].shape[1]  # max_consider_requests
        n_node = nodes["n_courier"].shape[1]
        device = nodes["n_courier"].device
        batch_arange = torch.arange(batch_size, device=device)
        courier_arange = torch.arange(n_courier, device=device)
        drone_arange = torch.arange(n_drone, device=device)

        # Calculate index boundaries
        courier_depot_end = n_courier
        courier_s1_pickup_end = n_courier + n_request
        courier_s3_pickup_end = n_courier + 2 * n_request
        courier_s1_delivery_end = n_courier + 3 * n_request
        courier_s3_delivery_end = n_courier + 4 * n_request
        drone_depot_end = n_courier + 4 * n_request + n_drone
        drone_s2_pickup_end = n_courier + 4 * n_request + n_drone + n_request
        drone_s2_delivery_end = n_courier + 4 * n_request + n_drone + 2 * n_request

        global_feature = self.global_proj_in(self.global_bn(torch.stack((
            _global["frame"].to(torch.float),
            _global["n_exist_requests"].to(torch.float),
        ), dim=1)))

        # Request validity mask (True = Valid)
        valid_mask = torch.arange(n_request, device=device) < _global["n_consider_requests"][:, None]

        # Stage masks (True = Active/Visible in this stage)
        cur_courier_request = _global["courier_request"]
        cur_drone_request = _global["drone_request"]
        stage1_mask = (cur_courier_request != 3).all(-2) & valid_mask
        stage2_mask = (cur_courier_request == 3).any(-2) & (cur_drone_request != 4).all(-2) & valid_mask
        stage3_mask = (cur_drone_request == 4).any(-2) & (cur_courier_request != 5).all(-2) & valid_mask
        
        # Filter to retain only nodes visible in each stage
        all_nodes = torch.cat((
            couriers["start_node"],  # [0, n_courier)
            requests["from"].masked_fill(~stage1_mask, 0),  # [n_courier, n_courier + n_request)
            requests["station2"].masked_fill(~stage3_mask, 0),  # [n_courier + n_request, n_courier + 2*n_request)
            requests["station1"].masked_fill(~stage1_mask, 0),  # [n_courier + 2*n_request, n_courier + 3*n_request)
            requests["to"].masked_fill(~stage3_mask, 0),  # [n_courier + 3*n_request, n_courier + 4*n_request)
            drones["start_node"],  # [n_courier + 4*n_request, n_courier + 4*n_request + n_drone)
            requests["station1"].masked_fill(~stage2_mask, 0),  # [n_courier + 4*n_request + n_drone, n_courier + 4*n_request + n_drone + n_request)
            requests["station2"].masked_fill(~stage2_mask, 0),  # [n_courier + 4*n_request + n_drone + n_request, n_courier + 4*n_request + n_drone + 2*n_request)
        ), dim=1)  # [batch_size, n_courier + 4*n_request + n_drone + 2*n_request]

        # Get node coordinates
        all_nodes_coord = nodes["coord"].gather(1, all_nodes.unsqueeze(-1).expand(-1, -1, 2))
        # Build request features
        # Divided by stages: Stage 1 - from, Stage 2 - station1, Stage 3 - to
        def build_request_feature(coord_source):
            coords = nodes["coord"].gather(1, requests[coord_source][:, :, None].expand(-1, -1, 2))
            return self.request_proj_in(torch.stack((
                requests["value"].to(torch.float),
                requests["volumn"].to(torch.float),
                coords[:, :, 0].to(torch.float),
                coords[:, :, -1].to(torch.float),
            ), dim=1).transpose(-1, -2))

        requests_feature_s1 = build_request_feature("from")
        requests_feature_s2 = build_request_feature("station1")
        requests_feature_s3 = build_request_feature("to")
        # Create initial dense embeddings for each node segment (excluding coordinates)
        # Depot initial features are DEPOT of the corresponding vehicle type
        # Pickup = original embedding + request feature + delivery feature
        # Delivery = original embedding + request feature
        node_segments = [
            # Courier segments
            ("courier_depot", self.COURIER_DEPOT, self.courier_depot_proj_in, n_courier, None, None),
            ("courier_s1_pickup", self.COURIER_STAGE1_PICKUP, self.courier_stage1_pickup_proj_in, n_request, requests_feature_s1, "courier_s1_delivery"),
            ("courier_s3_pickup", self.COURIER_STAGE3_PICKUP, self.courier_stage3_pickup_proj_in, n_request, requests_feature_s3, "courier_s3_delivery"),
            ("courier_s1_delivery", self.COURIER_STAGE1_DELIVERY, self.courier_stage1_delivery_proj_in, n_request, requests_feature_s1, None),
            ("courier_s3_delivery", self.COURIER_STAGE3_DELIVERY, self.courier_stage3_delivery_proj_in, n_request, requests_feature_s3, None),
            # Drone segments
            ("drone_depot", self.DRONE_DEPOT, self.drone_depot_proj_in, n_drone, None, None),
            ("drone_s2_pickup", self.DRONE_STAGE2_PICKUP, self.drone_stage2_pickup_proj_in, n_request, requests_feature_s2, "drone_s2_delivery"),
            ("drone_s2_delivery", self.DRONE_STAGE2_DELIVERY, self.drone_stage2_delivery_proj_in, n_request, requests_feature_s2, None),
        ]
        
        # First build all delivery dense embeddings (since pickup requires them)
        delivery_dense_dict = {}
        for name, type_emb, proj_in, n_nodes, req_feature, add_delivery_dense_name in node_segments:
            if "delivery" in name:
                # This is a delivery node, build dense embedding first
                dense = type_emb[None, None].expand(batch_size, n_nodes, -1)
                if req_feature is not None:
                    dense = dense + req_feature
                delivery_dense_dict[name] = dense
        
        # Build embeddings for all node segments
        all_nodes_embd_pre_list = []
        for name, type_emb, proj_in, n_nodes, req_feature, add_delivery_dense_name in node_segments:
            dense = type_emb[None, None].expand(batch_size, n_nodes, -1)
            if req_feature is not None:
                dense = dense + req_feature
            if add_delivery_dense_name is not None:
                # Pickup nodes need to add corresponding delivery dense embeddings
                dense = dense + delivery_dense_dict[add_delivery_dense_name]
            embd_pre = proj_in(dense)
            all_nodes_embd_pre_list.append(embd_pre)
        
        # Concatenate all node embeddings
        all_nodes_embd_pre = torch.cat(all_nodes_embd_pre_list, dim=1)
        # Add global_feature and coord_proj_in (in the order of original code)
        all_nodes_embd_pre = all_nodes_embd_pre + global_feature[:, None, :] + self.coord_proj_in(all_nodes_coord.to(torch.float))
        rel_mat = None
        # Build mask (True = Visible for encoder，False = Inactive/Padding)
        mask_segments = [
            torch.ones(batch_size, n_courier, dtype=torch.bool, device=device),  # courier depot (always visible)
            stage1_mask,  # courier s1 pickup (True = visible in stage1)
            stage3_mask,  # courier s3 pickup (True = visible in stage3)
            stage1_mask,  # courier s1 delivery (True = visible in stage1)
            stage3_mask,  # courier s3 delivery (True = visible in stage3)
            torch.ones(batch_size, n_drone, dtype=torch.bool, device=device),  # drone depot (always visible)
            stage2_mask,  # drone s2 pickup (True = visible in stage2)
            stage2_mask,  # drone s2 delivery (True = visible in stage2)
        ]
        all_nodes_mask = torch.cat(mask_segments, dim=1)
        # Unified encoding
        all_nodes_embd = self.encoder.forward(all_nodes_embd_pre, all_nodes_mask, rel_mat)
        # Index splitting (split after encoder)
        courier_depot_embd = all_nodes_embd[:, :courier_depot_end]
        courier_s1_pickup_embd = all_nodes_embd[:, courier_depot_end:courier_s1_pickup_end]
        courier_s3_pickup_embd = all_nodes_embd[:, courier_s1_pickup_end:courier_s3_pickup_end]
        courier_s1_delivery_embd = all_nodes_embd[:, courier_s3_pickup_end:courier_s1_delivery_end]
        courier_s3_delivery_embd = all_nodes_embd[:, courier_s1_delivery_end:courier_s3_delivery_end]
        drone_depot_embd = all_nodes_embd[:, courier_s3_delivery_end:drone_depot_end]
        drone_s2_pickup_embd = all_nodes_embd[:, drone_depot_end:drone_s2_pickup_end]
        drone_s2_delivery_embd = all_nodes_embd[:, drone_s2_pickup_end:drone_s2_delivery_end]

        # Re-pair after encoder according to original code logic: pickup = pickup + delivery
        courier_s1_pickup_embd = courier_s1_pickup_embd + courier_s1_delivery_embd
        courier_s3_pickup_embd = courier_s3_pickup_embd + courier_s3_delivery_embd
        drone_s2_pickup_embd = drone_s2_pickup_embd + drone_s2_delivery_embd

        # Calculate global_embd and values
        # Aggregate only active nodes
        global_embd_nodes = torch.cat((
            courier_depot_embd,
            courier_s1_pickup_embd.masked_fill(~stage1_mask[..., None], 0),
            courier_s3_pickup_embd.masked_fill(~stage3_mask[..., None], 0),
            courier_s1_delivery_embd.masked_fill(~stage1_mask[..., None], 0),
            courier_s3_delivery_embd.masked_fill(~stage3_mask[..., None], 0),
            drone_depot_embd,
            drone_s2_pickup_embd.masked_fill(~stage2_mask[..., None], 0),
            drone_s2_delivery_embd.masked_fill(~stage2_mask[..., None], 0),
        ), dim=1)
        
        # Calculate total number of visible nodes
        n_visible_nodes = (
            n_courier + n_drone +
            stage1_mask.sum(-1) * 2 + # s1 pickup + s1 delivery
            stage3_mask.sum(-1) * 2 + # s3 pickup + s3 delivery
            stage2_mask.sum(-1) * 2   # s2 pickup + s2 delivery
        ).float()[:, None]
        global_embd = global_embd_nodes.sum(1) / (n_visible_nodes + 1e-6)
        values = self.value_linear(global_embd).squeeze(-1)
        if only_critic:
            return values
        ###################################Start Decoding###################################

        # Node representations used for Courier decoding
        courier_nodes_embd = torch.cat((courier_depot_embd, courier_s1_pickup_embd, courier_s3_pickup_embd, courier_s1_delivery_embd, courier_s3_delivery_embd,), dim=1)
        # Build decodable node sequence
        courier_decode_nodes_embd = torch.cat((courier_s1_pickup_embd, courier_s3_pickup_embd, courier_s1_delivery_embd, courier_s3_delivery_embd, ), dim=1)  # [batch_size, 4*n_request, n_embd]
        
        # Build courier decoding input
        courier_task_target = couriers["task_target"]
        courier_target_embd = courier_nodes_embd[batch_arange[:, None], courier_task_target]
        invalid_task_mask = (courier_task_target < 0) | (courier_task_target >= courier_s3_delivery_end)
        courier_target_embd[invalid_task_mask] = self.NO_TARGET

        courier_decode_mask = couriers["time_left"] == 0
        courier_space = couriers["space"]
        comm_courier = torch.cat((courier_space, courier_target_embd.flatten(1)), -1)
        h_courier = torch.cat((
            courier_target_embd,
            courier_space[:, :, None],
            global_embd[:, None, :].expand(-1, n_courier, -1),
            comm_courier[:, None, :].expand(-1, n_courier, -1)
        ), -1)
        g_courier = self.cross_attn.forward(
            self.courier_proj_h(h_courier),
            courier_decode_nodes_embd, 
            courier_decode_nodes_embd
        )
        q_courier = self.to_q_g(g_courier)
        k_courier = self.to_k_g(courier_decode_nodes_embd)
        # Sampling and fleet handler
        D = 10
        d = self.n_embd / self.n_head
        k_courier_with_noaction = torch.cat((k_courier, self.NO_ACTION.expand(batch_size, 1, -1)), 1)
        u_courier = D * (q_courier @ k_courier_with_noaction.transpose(-1, -2) / d ** 0.5).tanh()
        # Update cur_courier_request status
        cur_courier_request = _global["courier_request"].clone()
        cur_courier_space = courier_space.clone()
        assert ((cur_courier_request == 1).sum(-1) <= 1).all()
        # Handle cases where pickup point has been reached
        cur_pickup_mask_courier = (couriers["time_left"] == 0) & ((cur_courier_request == 1).sum(-1) == 1)
        if cur_pickup_mask_courier.any():
            pickup_batch_courier, pickup_courier = torch.where(cur_pickup_mask_courier)
            courier_pickup_request = (cur_courier_request[pickup_batch_courier, pickup_courier] == 1).nonzero(as_tuple=True)[1]
            cur_courier_request[pickup_batch_courier, pickup_courier, courier_pickup_request] = 2
            cur_courier_space[pickup_batch_courier, pickup_courier] -= requests["volumn"][pickup_batch_courier, courier_pickup_request]
    
        # Build courier mask
        courier_mask = torch.ones(batch_size, n_courier, 4 * n_request, device=device, dtype=torch.bool)
        with torch.no_grad():
            dist_c_rows = _global["node_node"].to(torch.float).gather(1, couriers["target"].unsqueeze(-1).expand(-1, -1, n_node)) # [B, N_c, N_node]
            dist_s1 = dist_c_rows.gather(2, requests["from"].unsqueeze(1).expand(-1, n_courier, -1))
            dist_s1_masked = dist_s1.masked_fill(~stage1_mask[:, None, :], float('inf'))
            _, topk_s1_idx = dist_s1_masked.topk(min(3, n_request), dim=-1, largest=False)
            dist_mask_s1 = torch.zeros_like(dist_s1, dtype=torch.bool).scatter_(2, topk_s1_idx, True)
            dist_mask_s1 = dist_mask_s1 & stage1_mask[:, None, :]
    
            dist_s3 = dist_c_rows.gather(2, requests["station2"].unsqueeze(1).expand(-1, n_courier, -1))
            dist_s3_masked = dist_s3.masked_fill(~stage3_mask[:, None, :], float('inf'))
            _, topk_s3_idx = dist_s3_masked.topk(min(3, n_request), dim=-1, largest=False)
            dist_mask_s3 = torch.zeros_like(dist_s3, dtype=torch.bool).scatter_(2, topk_s3_idx, True)
            dist_mask_s3 = dist_mask_s3 & stage3_mask[:, None, :]

        s1_pickup_mask = (
            (cur_courier_request == 0).all(1)[:, None, :] &  # no_relation
            (courier_decode_mask[:, :, None]) &  # vehicle可解码
            (cur_courier_space[:, :, None] >= requests["volumn"][:, None, :]) &  # 有空间
            stage1_mask[:, None, :] 
            & dist_mask_s1
        )
        courier_mask[:, :, :n_request] = s1_pickup_mask
        # This part is a bit complex
        s3_pickup_mask = (
            ((cur_courier_request == 3).sum(1) == 1)[:, None, :] &  # 恰好有一个courier关系为3
            ((cur_courier_request == 0) | (cur_courier_request == 3)).all(1)[:, None, :] &  # 除此之外全为0
            (cur_courier_request != 3) & # 防止同个courier执行stage1和stage3
            (courier_decode_mask[:, :, None]) &  # vehicle可解码
            (cur_courier_space[:, :, None] >= requests["volumn"][:, None, :]) &  # 有空间
            stage3_mask[:, None, :] 
            & dist_mask_s3
        )
        courier_mask[:, :, n_request:2*n_request] = s3_pickup_mask

        # Stage1 delivery mask: request is in to_delivery state and courier's target == station1, courier is carrying the request
        s1_delivery_mask = (
            (cur_courier_request == 2) &
            (courier_decode_mask[:, :, None]) &
            stage1_mask[:, None, :]
        )
        courier_mask[:, :, 2*n_request:3*n_request] = s1_delivery_mask

        # Stage3 delivery mask: request is in to_delivery state and courier's target == to node, courier is carrying the request
        s3_delivery_mask = (
            (cur_courier_request == 2)&
            (courier_decode_mask[:, :, None]) & 
            stage3_mask[:, None, :] 
        )
        courier_mask[:, :, 3*n_request:4*n_request] = s3_delivery_mask
        
        u_courier[:, :, :-1].masked_fill_(~courier_mask, -torch.inf)
        u_courier[..., -1].masked_fill_((u_courier[:, :, :-1] == -torch.inf).all(-1), 1)
        p_courier = F.softmax(u_courier, -1)
        # Must perform tasks when available
        p_courier = p_courier.masked_fill(torch.cat((
            torch.zeros_like(p_courier, dtype=torch.bool)[..., :-1],
            (p_courier[..., :-1].sum(-1) != 0).unsqueeze(-1)
        ), dim=-1), 0)

        cate_courier = Categorical(probs=p_courier)
        is_decoding = input_actions is None
        if is_decoding:
            sampled_new_task_target_courier = cate_courier.sample()
            new_task_target_courier = sampled_new_task_target_courier.clone()
            cur_courier_request_handler = cur_courier_request.clone()
            for idx in range(n_courier):
                # Handle stage1 pickup conflict detection
                s1_pickup_mask = courier_decode_mask[:, idx] & (new_task_target_courier[:, idx] < n_request)
                s1_pickup_batch = batch_arange[s1_pickup_mask]
                s1_pickup_request = new_task_target_courier[:, idx][s1_pickup_mask]
                no_conflict_s1 = (cur_courier_request_handler[s1_pickup_batch, :, s1_pickup_request] == 0).all(-1)
                new_task_target_courier[torch.where(s1_pickup_mask)[0][~no_conflict_s1], idx] = 4 * n_request  # NO_ACTION
                cur_courier_request_handler[s1_pickup_batch[no_conflict_s1], idx, s1_pickup_request[no_conflict_s1]] = 1
                
                # Handle stage3 pickup conflict detection
                s3_pickup_mask = courier_decode_mask[:, idx] & (new_task_target_courier[:, idx] >= n_request) & (new_task_target_courier[:, idx] < 2 * n_request)
                s3_pickup_batch = batch_arange[s3_pickup_mask]
                s3_pickup_request = new_task_target_courier[:, idx][s3_pickup_mask] - n_request  # 转换为request索引
                conflict_mask = (cur_courier_request_handler[s3_pickup_batch, :, s3_pickup_request] == 1) | \
                                (cur_courier_request_handler[s3_pickup_batch, :, s3_pickup_request] == 2)
                no_conflict_s3 = ~conflict_mask.any(-1)
                new_task_target_courier[torch.where(s3_pickup_mask)[0][~no_conflict_s3], idx] = 4 * n_request  # NO_ACTION
                cur_courier_request_handler[s3_pickup_batch[no_conflict_s3], idx, s3_pickup_request[no_conflict_s3]] = 1

            # Decode courier actions
            courier_to = couriers["target"].clone()
            request_courier = torch.full([batch_size, n_request], n_courier, dtype=torch.int64, device=device)
            if cur_pickup_mask_courier.any():
                request_courier[pickup_batch_courier, courier_pickup_request] = pickup_courier
            
            # Stage1 pickup
            s1_pickup_mask_action = (new_task_target_courier < n_request) & courier_decode_mask
            courier_to[s1_pickup_mask_action] = requests["from"][batch_arange[:, None].masked_select(s1_pickup_mask_action), new_task_target_courier[s1_pickup_mask_action]]
            # Stage3 pickup
            s3_pickup_mask_action = (new_task_target_courier >= n_request) & (new_task_target_courier < 2 * n_request) & courier_decode_mask
            courier_to[s3_pickup_mask_action] = requests["station2"][batch_arange[:, None].masked_select(s3_pickup_mask_action), new_task_target_courier[s3_pickup_mask_action] - n_request]
            # Stage1 delivery
            s1_delivery_mask_action = (new_task_target_courier >= 2 * n_request) & (new_task_target_courier < 3 * n_request) & courier_decode_mask
            courier_to[s1_delivery_mask_action] = requests["station1"][batch_arange[:, None].masked_select(s1_delivery_mask_action), new_task_target_courier[s1_delivery_mask_action] - 2 * n_request]
            # Stage3 delivery
            s3_delivery_mask_action = (new_task_target_courier >= 3 * n_request) & (new_task_target_courier < 4 * n_request) & courier_decode_mask
            courier_to[s3_delivery_mask_action] = requests["to"][batch_arange[:, None].masked_select(s3_delivery_mask_action), new_task_target_courier[s3_delivery_mask_action] - 3 * n_request]
            # decode courier preassign
            courier_preassign = torch.full([batch_size, n_request], n_courier, dtype=torch.int64, device=device)
            courier_preassign[batch_arange[:, None].masked_select(s1_pickup_mask_action), new_task_target_courier[s1_pickup_mask_action]] = courier_arange.masked_select(s1_pickup_mask_action)
            courier_preassign[batch_arange[:, None].masked_select(s3_pickup_mask_action), new_task_target_courier[s3_pickup_mask_action] - n_request] = courier_arange.masked_select(s3_pickup_mask_action)
        else:
            new_task_target_courier = input_actions["new_task_target_courier"]
            courier_to = input_actions["courier"]
            request_courier = input_actions["request_courier"]
            sampled_new_task_target_courier = input_actions["sampled_new_task_target_courier"]
            courier_preassign = input_actions["courier_preassign"]
        log_prob_courier = cate_courier.log_prob(sampled_new_task_target_courier).sum(-1)
        entropy_courier = cate_courier.entropy().sum(-1)
        ###################################Decode drone actions###################################
        # Build decodable node sequence (using paired pickup and delivery)
        drone_decode_nodes_embd = torch.cat((
            drone_s2_pickup_embd,  # Already paired: pickup + delivery
            drone_s2_delivery_embd,
        ), dim=1)  # [batch_size, 2*n_request, n_embd]

        # Build drone decoding input
        drone_task_target = drones["task_target"]
        # Convert drone_task_target from full sequence index to relative index within drone segment
        drone_task_target_relative = drone_task_target - courier_s3_delivery_end
        # Build drone_nodes_embd for indexing (including depot, pickup, delivery)
        drone_nodes_embd = torch.cat((
            drone_depot_embd,
            drone_s2_pickup_embd,  # Paired pickup
            drone_s2_delivery_embd,
        ), dim=1)
        drone_target_embd = drone_nodes_embd[batch_arange[:, None], drone_task_target_relative]
        invalid_drone_task_mask = (drone_task_target < courier_s3_delivery_end) | (drone_task_target >= drone_s2_delivery_end)
        drone_target_embd[invalid_drone_task_mask] = self.NO_TARGET

        drone_decode_mask = drones["time_left"] == 0
        drone_space = drones["space"]
        comm_drone = torch.cat((drone_space, drone_target_embd.flatten(1)), -1)
        h_drone = torch.cat((
            drone_target_embd,
            drone_space[:, :, None],
            global_embd[:, None, :].expand(-1, n_drone, -1),
            comm_drone[:, None, :].expand(-1, n_drone, -1)
        ), -1)
        g_drone = self.cross_attn.forward(
            self.drone_proj_h(h_drone),
            drone_decode_nodes_embd,
            drone_decode_nodes_embd
        )
        q_drone = self.to_q_g(g_drone)
        k_drone = self.to_k_g(drone_decode_nodes_embd)

        # Update cur_drone_request status
        cur_drone_request = _global["drone_request"].clone()
        cur_drone_space = drone_space.clone()
        # Handle cases where pickup point has been reached
        assert ((cur_drone_request == 1).sum(-1) <= 1).all()
        cur_pickup_mask_drone = (drones["time_left"] == 0) & ((cur_drone_request == 1).sum(-1) == 1)
        if cur_pickup_mask_drone.any():
            pre_pickup_batch_drone, pre_pickup_drone = torch.where(cur_pickup_mask_drone)
            pre_pickup_request_drone = (cur_drone_request[pre_pickup_batch_drone, pre_pickup_drone] == 1).nonzero(as_tuple=True)[1]
            cur_drone_request[pre_pickup_batch_drone, pre_pickup_drone, pre_pickup_request_drone] = 2
            cur_drone_space[pre_pickup_batch_drone, pre_pickup_drone] -= requests["volumn"][pre_pickup_batch_drone, pre_pickup_request_drone]

        # Decode drone actions (ensure target is a station node)
        drone_to = drones["target"].clone()
        request_drone = torch.full([batch_size, n_request], n_drone, dtype=torch.int64, device=device)
        drone_preassign = torch.full([batch_size, n_request], n_drone, dtype=torch.int64, device=device)
        
        # Drones that have accepted orders go directly to destination
        cur_drone_need_decode = drones["time_left"] == 0
        has_request_stage2 = (cur_drone_request == 2).any(dim=-1) # [B, D]
        special_drone_idx = (cur_drone_need_decode & has_request_stage2).nonzero(as_tuple=True)
        if len(special_drone_idx[0]) > 0:
            request_idx = (cur_drone_request[special_drone_idx] == 2).int().argmax(dim=-1)
            drone_to[special_drone_idx] = requests["station2"][special_drone_idx[0], request_idx]
            # In this case, drone_decode_mask should be False in subsequent calculations to avoid duplicate decisions
        
        # Build drone mask
        drone_mask = torch.ones(batch_size, n_drone, 2 * n_request + 1, device=device, dtype=torch.bool)
        drone_mask[has_request_stage2, :] = False

        with torch.no_grad():
            dist_d_rows = _global["node_node"].to(torch.float).gather(1, drones["target"].unsqueeze(-1).expand(-1, -1, n_node)) # [B, N_d, N_node]
            station1_nodes_count = torch.zeros(batch_size, n_node + 1, dtype=torch.int, device=device)  # 多一个位置来存放无效索引
            selected_node_mask = torch.zeros_like(dist_d_rows, dtype=torch.bool)  # [B, N_d, N_node]

            station1_nodes_count.scatter_add_(1, torch.where(stage2_mask, requests["station1"], n_node), stage2_mask.int())  # [B, N_node+1]
            station1_nodes_in_stage2 = station1_nodes_count[:, :n_node] > 0  # [B, N_node]
            
            dist_to_station1_nodes = dist_d_rows.masked_fill(~station1_nodes_in_stage2[:, None, :], float('inf'))  # [B, N_d, N_node]
            _, topk_node_idx = dist_to_station1_nodes.topk(2, dim=-1, largest=False)  # [B, N_d, topk]
            selected_node_mask = selected_node_mask.scatter_(2, topk_node_idx, True) & station1_nodes_in_stage2[:, None, :]
            dist_mask_s2 = selected_node_mask.gather(2, requests["station1"].unsqueeze(1).expand(-1, n_drone, -1))  # [B, N_d, N_request]
            dist_mask_s2 = dist_mask_s2 & stage2_mask[:, None, :]

        s2_pickup_mask = (
            (cur_drone_request == 0).all(-2)[:, None, :] &  # no_relation (from drone perspective)
            (cur_drone_space[:, :, None] >= requests["volumn"][:, None, :]) &  # Has enough space
            (drone_decode_mask[:, :, None]) &  # vehicle is decodable
            stage2_mask[:, None, :] 
            & dist_mask_s2  # valid request & distance filtering
        )
        drone_mask[:, :, :n_request] = s2_pickup_mask
        drone_mask[:, :, n_request:2*n_request] = False

        # Sampling and fleet handler
        k_drone_with_noaction = torch.cat((k_drone, self.NO_ACTION.expand(batch_size, 1, -1)), 1)
        u_drone = D * (q_drone @ k_drone_with_noaction.transpose(-1, -2) / d ** 0.5).tanh()
        u_drone[:, :, :-1].masked_fill_(~drone_mask[:, :, :-1], -torch.inf)
        u_drone[:, :, -1].masked_fill_((u_drone[:, :, :-1] == -torch.inf).all(-1), 1)
        p_drone = F.softmax(u_drone, -1)
        p_drone = p_drone.masked_fill(torch.cat((
            torch.zeros_like(p_drone, dtype=torch.bool)[..., :-1],
            (p_drone[..., :-1].sum(-1) != 0).unsqueeze(-1)
        ), dim=-1), 0)
        cate_drone = Categorical(probs=p_drone)

        if is_decoding:
            sampled_new_task_target_drone = cate_drone.sample()
            
            new_task_target_drone = sampled_new_task_target_drone.clone()
            cur_drone_request_handler = cur_drone_request.clone()
            for idx in range(n_drone):
                vehicle_mask = drone_decode_mask[:, idx]
                if not vehicle_mask.any():
                    continue
                pickup_mask_drone = vehicle_mask & (new_task_target_drone[:, idx] < n_request)
                pickup_batch_drone = batch_arange[pickup_mask_drone]
                pickup_request_drone = new_task_target_drone[pickup_batch_drone, idx]
                # Conflict detection: ensure the same request is not picked up by multiple drones simultaneously
                no_conflict = (cur_drone_request_handler[pickup_batch_drone, :, pickup_request_drone] == 0).all(-1)
                new_task_target_drone[torch.where(pickup_mask_drone)[0][~no_conflict]] = 2 * n_request  # NO_ACTION
                cur_drone_request_handler[pickup_batch_drone[no_conflict], idx, pickup_request_drone[no_conflict]] = 1

            # Decode drone actions (ensure target is a station node)
            if cur_pickup_mask_drone.any():
                request_drone[pre_pickup_batch_drone, pre_pickup_request_drone] = pre_pickup_drone
            
            # Stage2 pickup
            s2_pickup_mask_action = (new_task_target_drone < n_request) & drone_decode_mask & (~has_request_stage2)
            drone_to[s2_pickup_mask_action] = requests["station1"][
                batch_arange[:, None].masked_select(s2_pickup_mask_action),
                new_task_target_drone[s2_pickup_mask_action]
            ]
            # Stage2 pickup
            drone_preassign[batch_arange[:, None].masked_select(s2_pickup_mask_action), new_task_target_drone[s2_pickup_mask_action]] = drone_arange.masked_select(s2_pickup_mask_action)
            log_prob_drone = cate_drone.log_prob(sampled_new_task_target_drone).sum(-1)
            entropy_drone = cate_drone.entropy().sum(-1)
            # Map drone's task_target back to full sequence index
            new_task_target_drone_full = new_task_target_drone + courier_s3_delivery_end
            sampled_new_task_target_drone_full = sampled_new_task_target_drone + courier_s3_delivery_end
        else:
            # task_target in input_actions is already full sequence index, need to convert to relative index for log_prob calculation
            new_task_target_drone_full = input_actions["new_task_target_drone"]
            sampled_new_task_target_drone_full = input_actions["sampled_new_task_target_drone"]
            drone_to = input_actions["drone"]
            request_drone = input_actions["request_drone"]
            drone_preassign = input_actions["drone_preassign"]
            # Convert to relative index for log_prob calculation
            sampled_new_task_target_drone = sampled_new_task_target_drone_full - courier_s3_delivery_end
            log_prob_drone = cate_drone.log_prob(sampled_new_task_target_drone).sum(-1)
            entropy_drone = cate_drone.entropy().sum(-1)

        # Merge outputs
        station1_action = obs["requests"]["station1"]
        station2_action = obs["requests"]["station2"]

        return {
            "station1": station1_action,
            "station2": station2_action,
            "request_courier": request_courier,
            "request_drone": request_drone,
            "courier": courier_to,
            "drone": drone_to,
            "new_task_target_courier": new_task_target_courier+n_courier,
            "new_task_target_drone": new_task_target_drone_full+n_drone,
            "courier_preassign": courier_preassign,
            "drone_preassign": drone_preassign,
            "sampled_new_task_target_courier": sampled_new_task_target_courier,
            "sampled_new_task_target_drone": sampled_new_task_target_drone_full,
        }, log_prob_courier + log_prob_drone, entropy_courier + entropy_drone, values

MAPDP.register_algo_specific()
