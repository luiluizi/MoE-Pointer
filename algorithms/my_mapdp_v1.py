import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical

from .models import BatchNorm, Encoder, MLP, Attention, IterMixin
from envs.mvdpdp.env import obs_func_map, trans_func_map, DroneTransferEnv


class MAPDP(nn.Module, IterMixin):

    @staticmethod
    def obs_func(env: DroneTransferEnv, obs, unobs):
        if env._global["frame"] == 0:
            # 初始化couriers
            if "start_node" not in env.couriers:
                env.couriers["start_node"] = env.couriers["target"].clone()
            else:
                env.couriers["start_node"] = env.couriers["target"].clone()
            if "task_target" not in env.couriers:
                env.couriers["task_target"] = torch.full(
                    [env.batch_size, env.n_courier], -1, 
                    dtype=torch.int64, device=env.device
                )
            
            # 初始化drones
            if "start_node" not in env.drones:
                env.drones["start_node"] = env.drones["target"].clone()
            else:
                env.drones["start_node"] = env.drones["target"].clone()
            if "task_target" not in env.drones:
                env.drones["task_target"] = torch.full(
                    [env.batch_size, env.n_drone], -1,
                    dtype=torch.int64, device=env.device
                )
        
        # 将task_target添加到obs中
        if "task_target" not in obs["couriers"]:
            obs["couriers"]["task_target"] = env.couriers["task_target"].clone()
        else:
            obs["couriers"]["task_target"] = env.couriers["task_target"].clone()
        if "start_node" not in obs["couriers"]:
            obs["couriers"]["start_node"] = env.couriers["start_node"].clone()
        else:
            obs["couriers"]["start_node"] = env.couriers["start_node"].clone()
        
        if "task_target" not in obs["drones"]:
            obs["drones"]["task_target"] = env.drones["task_target"].clone()
        else:
            obs["drones"]["task_target"] = env.drones["task_target"].clone()
        if "start_node" not in obs["drones"]:
            obs["drones"]["start_node"] = env.drones["start_node"].clone()
        else:
            obs["drones"]["start_node"] = env.drones["start_node"].clone()
    
    @staticmethod
    def trans_func(env: DroneTransferEnv, actions):
        # 处理courier的new_task_target（统一处理Stage1和Stage3）
        if "courier_new_task_target" in actions:
            courier_new_task_target = actions["courier_new_task_target"]
            courier_new_task_mask = courier_new_task_target >= 0
            env.couriers["task_target"][courier_new_task_mask] = courier_new_task_target[courier_new_task_mask]
        
        # 处理drone的new_task_target（Stage2）
        if "drone_new_task_target" in actions:
            drone_new_task_target = actions["drone_new_task_target"]
            drone_new_task_mask = drone_new_task_target >= 0
            env.drones["task_target"][drone_new_task_mask] = drone_new_task_target[drone_new_task_mask]

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
        
        n_courier = env_args["n_courier"]
        n_drone = env_args["n_drone"]

        # BatchNorm should be turn off when rollout/evaluate
        self.global_bn = BatchNorm(2)
        self.global_proj_in = nn.Linear(2, n_embd)
        self.request_proj_in = nn.Linear(4, n_embd)
        self.request_request_dist_proj_in = MLP(1, n_embd, rel_dim)
        self.value_linear = MLP(n_embd, n_embd, n_embd, 1)
        self.coord_proj_in = nn.Linear(2, n_embd)

        # Courier节点embedding（统一处理Stage1和Stage3）
        self.depot_courier_proj_in = torch.nn.Linear(n_embd, n_embd)
        self.from_proj_in = torch.nn.Linear(n_embd, n_embd)  # Stage1的from和Stage3的station2
        self.station1_proj_in = torch.nn.Linear(n_embd, n_embd)  # Stage1的目标节点
        self.to_proj_in = torch.nn.Linear(n_embd, n_embd)  # Stage3的目标节点
        
        # Drone节点embedding（Stage2）
        self.depot_drone_proj_in = torch.nn.Linear(n_embd, n_embd)
        self.station1_drone_proj_in = torch.nn.Linear(n_embd, n_embd)
        self.station2_drone_proj_in = torch.nn.Linear(n_embd, n_embd)
        
        # Embedding参数
        self.DEPOT_COURIER = nn.Parameter(torch.randn(n_embd))
        self.FROM = nn.Parameter(torch.randn(n_embd))  # Stage1的from和Stage3的station2
        self.STATION1 = nn.Parameter(torch.randn(n_embd))  # Stage1的目标节点
        self.TO = nn.Parameter(torch.randn(n_embd))  # Stage3的目标节点
        
        self.DEPOT_DRONE = nn.Parameter(torch.randn(n_embd))
        self.STATION1_DRONE = nn.Parameter(torch.randn(n_embd))
        self.STATION2_DRONE = nn.Parameter(torch.randn(n_embd))
        
        self.NO_TARGET = nn.Parameter(torch.randn(n_embd))
        self.NO_ACTION = nn.Parameter(torch.randn(n_embd))
        self.NO_ACTION_DRONE = nn.Parameter(torch.randn(n_embd))

        self.encoder = Encoder(n_enc_bloc, n_embd, n_head, qkfeat_dim=rel_dim)
        self.encoder_drone = Encoder(n_enc_bloc, n_embd, n_head, qkfeat_dim=rel_dim)

        # Courier的proj_h（统一处理Stage1和Stage3）
        self.proj_h = nn.Linear(n_courier*(1+n_embd)+2 * n_embd + 1, n_embd)
        self.cross_attn = Attention(n_embd, n_head)
        self.to_q_g = nn.Linear(n_embd, n_embd)  # 共享决策头
        self.to_k_g = nn.Linear(n_embd, n_embd)  # 共享决策头
        
        # Drone的proj_h（独立处理Stage2）
        self.proj_h_drone = nn.Linear(n_drone*(1+n_embd)+2 * n_embd + 1, n_embd)
        self.cross_attn_drone = Attention(n_embd, n_head)
        self.to_q_g_drone = nn.Linear(n_embd, n_embd)
        self.to_k_g_drone = nn.Linear(n_embd, n_embd)

    def _assign_stations_heuristic(self, requests, _global, requests_mask, device):
        """启发式分配station1和station2"""
        batch_size = requests["from"].shape[0]
        n_request = requests["from"].shape[1]
        n_node = _global["node_node"].shape[1]
        
        station1_action = torch.full([batch_size, n_request], n_node, dtype=torch.int64, device=device)
        station2_action = torch.full([batch_size, n_request], n_node, dtype=torch.int64, device=device)
        
        for b in range(batch_size):
            valid_requests = requests_mask[b]
            if not valid_requests.any():
                continue
                
            from_nodes = requests["from"][b, valid_requests]
            to_nodes = requests["to"][b, valid_requests]
            stations = _global["station_idx"][b, :-1]  # 排除padding
            
            if stations.numel() == 0:
                continue
            
            # 为每个请求找到最近的station作为station1
            for i, (from_node, to_node) in enumerate(zip(from_nodes, to_nodes)):
                if from_node >= n_node or to_node >= n_node:
                    continue
                dist_from = _global["node_node"][b, from_node, stations]
                if dist_from.numel() > 0:
                    station1_idx = stations[dist_from.argmin()]
                    station1_action[b, torch.where(valid_requests)[0][i]] = station1_idx
                    
                    # 从station1到to，找到最近的station作为station2
                    dist_to = _global["node_node"][b, stations, to_node]
                    if dist_to.numel() > 0:
                        station2_idx = stations[dist_to.argmin()]
                        station2_action[b, torch.where(valid_requests)[0][i]] = station2_idx
        
        return station1_action, station2_action

    def _build_courier_tasks(self, courier_request, drone_request, requests, requests_mask, n_stage1_tasks_max, device):
        """构建Courier统一任务池（Stage1和Stage3）"""
        batch_size = courier_request.shape[0]
        n_request = courier_request.shape[2]
        
        # Stage1任务：courier_request == 0（所有courier都是0）
        stage1_mask = (courier_request == 0).all(dim=1) & requests_mask  # [B, n_request]
        
        # Stage3任务：drone_request == 4 且 courier_request != 5
        drone_has_eq4 = (drone_request == 4).any(dim=1)  # [B, n_request]
        courier_has_eq5 = (courier_request == 5).any(dim=1)  # [B, n_request]
        stage3_mask = drone_has_eq4 & (~courier_has_eq5) & requests_mask  # [B, n_request]
        
        # 构建任务列表和映射
        task_to_request_map = []  # [B, n_tasks] -> request_idx
        task_type_map = []  # [B, n_tasks] -> 'stage1' or 'stage3'
        n_stage1_tasks = []
        
        for b in range(batch_size):
            stage1_requests = torch.where(stage1_mask[b])[0]
            stage3_requests = torch.where(stage3_mask[b])[0]
            
            n_s1 = min(stage1_requests.numel(), n_stage1_tasks_max)
            n_s3 = stage3_requests.numel()
            
            # Stage1任务索引：[0, n_s1)
            # Stage3任务索引：[n_s1, n_s1 + n_s3)
            task_map = torch.full([n_s1 + n_s3], -1, dtype=torch.int64, device=device)
            task_type = torch.full([n_s1 + n_s3], -1, dtype=torch.int64, device=device)
            
            # Stage1任务
            for i in range(n_s1):
                task_map[i] = stage1_requests[i]
                task_type[i] = 1  # 1表示Stage1
            
            # Stage3任务
            for i in range(n_s3):
                task_map[n_s1 + i] = stage3_requests[i]
                task_type[n_s1 + i] = 3  # 3表示Stage3
            
            task_to_request_map.append(task_map)
            task_type_map.append(task_type)
            n_stage1_tasks.append(n_s1)
        
        return task_to_request_map, task_type_map, n_stage1_tasks, stage1_mask, stage3_mask

    def _build_drone_tasks(self, courier_request, drone_request, requests, requests_mask, device):
        """构建Drone任务池（Stage2）"""
        batch_size = courier_request.shape[0]
        n_request = courier_request.shape[2]
        
        # Stage2任务：courier_request == 3 且 drone_request == 0
        courier_has_eq3 = (courier_request == 3).any(dim=1)  # [B, n_request]
        drone_has_eq0 = (drone_request == 0).all(dim=1)  # [B, n_request]
        stage2_mask = courier_has_eq3 & drone_has_eq0 & requests_mask  # [B, n_request]
        
        # 构建任务列表和映射
        task_to_request_map = []
        
        for b in range(batch_size):
            stage2_requests = torch.where(stage2_mask[b])[0]
            task_map = stage2_requests.clone()
            task_to_request_map.append(task_map)
        
        return task_to_request_map, stage2_mask

    def forward(self, obs, input_actions=None, deterministic=False, only_critic=False):
        """
        Note: 不要对输入进行 inplace 操作
        Note: max_visible_
        """
        nodes = obs["nodes"]
        couriers = obs["couriers"]
        drones = obs["drones"]
        requests = obs["requests"]
        _global = obs["global"]

        batch_size = _global["n_exist_requests"].shape[0]
        n_courier = couriers["capacity"].shape[1]
        n_drone = drones["space"].shape[1]
        n_request = requests["value"].shape[1]  # max_consider_requests
        n_node = nodes["n_courier"].shape[1]
        device = nodes["n_courier"].device
        batch_arange = torch.arange(batch_size, device=device)
        courier_arange = torch.arange(n_courier, device=device)
        drone_arange = torch.arange(n_drone, device=device)

        courier_decode_mask = couriers["time_left"] == 0
        drone_decode_mask = drones["time_left"] == 0
        
        requests_mask = torch.arange(n_request, device=device) < _global["n_consider_requests"][:, None]
        
        # Station分配启发式（在obs_func中应该已经分配，这里作为备用）
        # 如果requests中没有station1/station2，则使用启发式分配
        if "station1" not in requests or (requests["station1"] >= n_node).all():
            station1_action, station2_action = self._assign_stations_heuristic(
                requests, _global, requests_mask, device
            )
        else:
            station1_action = requests["station1"].clone()
            station2_action = requests["station2"].clone()

        global_feature = self.global_proj_in(self.global_bn(torch.stack((
            _global["frame"].to(torch.float),
            _global["n_exist_requests"].to(torch.float),
        ), dim=1)))

        # 构建任务列表
        courier_request = _global["courier_request"]
        drone_request = _global["drone_request"]
        
        # Courier任务池（统一Stage1和Stage3）
        max_courier_tasks = n_request * 2  # 最大任务数
        courier_task_map, courier_task_type, n_stage1_tasks_list, stage1_mask, stage3_mask = self._build_courier_tasks(
            courier_request, drone_request, requests, requests_mask, max_courier_tasks, device
        )
        
        # Drone任务池（Stage2）
        drone_task_map, stage2_mask = self._build_drone_tasks(
            courier_request, drone_request, requests, requests_mask, device
        )
        
        # 找到每个batch的最大任务数（用于padding）
        max_courier_tasks_batch = max([m.numel() for m in courier_task_map] + [1])
        max_drone_tasks_batch = max([m.numel() for m in drone_task_map] + [1])
        
        # 构建统一的Courier节点序列：[depot, stage1_from, stage1_station1, stage3_station2, stage3_to]
        # 需要为每个batch构建节点序列
        # 这里简化处理：使用所有可能的节点，通过mask控制可见性
        
        # 准备请求特征
        requests_coord = nodes["coord"].gather(1, requests["from"][:, :, None].expand(-1, -1, 2))
        requests_feature = self.request_proj_in(torch.stack((
            requests["value"].to(torch.float),
            requests["volumn"].to(torch.float),
            requests_coord[:, :, 0].to(torch.float),
            requests_coord[:, :, -1].to(torch.float),
        ), dim=1).transpose(-1, -2))
        
        # Courier节点编码（需要根据任务动态构建）
        # 简化版本：为所有可能的节点类型创建embedding
        courier_nodes_embd_list = []
        courier_nodes_mask_list = []
        
        for b in range(batch_size):
            # Depot节点
            depot_embd = self.DEPOT_COURIER[None, :].expand(n_courier, -1)
            
            # Stage1任务节点：from和station1
            stage1_tasks = courier_task_map[b][courier_task_type[b] == 1]
            n_s1 = stage1_tasks.numel()
            if n_s1 > 0:
                stage1_from_nodes = requests["from"][b, stage1_tasks]  # [n_s1]
                stage1_station1_nodes = station1_action[b, stage1_tasks]  # [n_s1]
                
                from_coord = nodes["coord"][b, stage1_from_nodes]  # [n_s1, 2]
                station1_coord = nodes["coord"][b, stage1_station1_nodes]  # [n_s1, 2]
                
                from_embd = self.from_proj_in(
                    self.FROM[None, :].expand(n_s1, -1) + 
                    requests_feature[b, stage1_tasks] +
                    self.coord_proj_in(from_coord)
                )
                station1_embd = self.station1_proj_in(
                    self.STATION1[None, :].expand(n_s1, -1) +
                    requests_feature[b, stage1_tasks] +
                    self.coord_proj_in(station1_coord)
                )
            else:
                from_embd = torch.empty([0, self.n_embd], device=device)
                station1_embd = torch.empty([0, self.n_embd], device=device)
            
            # Stage3任务节点：station2和to
            stage3_tasks = courier_task_map[b][courier_task_type[b] == 3]
            n_s3 = stage3_tasks.numel()
            if n_s3 > 0:
                stage3_station2_nodes = station2_action[b, stage3_tasks]  # [n_s3]
                stage3_to_nodes = requests["to"][b, stage3_tasks]  # [n_s3]
                
                station2_coord = nodes["coord"][b, stage3_station2_nodes]  # [n_s3, 2]
                to_coord = nodes["coord"][b, stage3_to_nodes]  # [n_s3, 2]
                
                station2_embd = self.from_proj_in(  # 使用FROM类型
                    self.FROM[None, :].expand(n_s3, -1) +
                    requests_feature[b, stage3_tasks] +
                    self.coord_proj_in(station2_coord)
                )
                to_embd = self.to_proj_in(
                    self.TO[None, :].expand(n_s3, -1) +
                    requests_feature[b, stage3_tasks] +
                    self.coord_proj_in(to_coord)
                )
            else:
                station2_embd = torch.empty([0, self.n_embd], device=device)
                to_embd = torch.empty([0, self.n_embd], device=device)
            
            # 合并节点序列：[depot, stage1_from, stage1_station1, stage3_station2, stage3_to]
            courier_nodes = torch.cat([
                depot_embd + global_feature[b, None, :],
                from_embd + global_feature[b, None, :],
                station1_embd + global_feature[b, None, :],
                station2_embd + global_feature[b, None, :],
                to_embd + global_feature[b, None, :]
            ], dim=0)  # [n_courier + n_s1*2 + n_s3*2, n_embd]
            
            # 创建mask
            n_courier_nodes = n_courier + n_s1 * 2 + n_s3 * 2
            courier_nodes_mask = torch.ones(n_courier_nodes, dtype=torch.bool, device=device)
            
            courier_nodes_embd_list.append(courier_nodes)
            courier_nodes_mask_list.append(courier_nodes_mask)
        
        # Padding到统一长度
        max_courier_nodes = max([e.shape[0] for e in courier_nodes_embd_list] + [n_courier])
        courier_nodes_embd_padded = torch.zeros(
            batch_size, max_courier_nodes, self.n_embd, device=device
        )
        courier_nodes_mask_padded = torch.zeros(
            batch_size, max_courier_nodes, dtype=torch.bool, device=device
        )
        
        for b in range(batch_size):
            n_nodes = courier_nodes_embd_list[b].shape[0]
            courier_nodes_embd_padded[b, :n_nodes] = courier_nodes_embd_list[b]
            courier_nodes_mask_padded[b, :n_nodes] = courier_nodes_mask_list[b]
        
        # 使用encoder处理
        courier_nodes_embd = self.encoder.forward(
            courier_nodes_embd_padded, courier_nodes_mask_padded, None
        )
        
        # Drone节点编码（类似处理）
        drone_nodes_embd_list = []
        drone_nodes_mask_list = []
        
        for b in range(batch_size):
            depot_drone_embd = self.DEPOT_DRONE[None, :].expand(n_drone, -1)
            
            stage2_tasks = drone_task_map[b]
            n_s2 = stage2_tasks.numel()
            if n_s2 > 0:
                stage2_station1_nodes = station1_action[b, stage2_tasks]
                stage2_station2_nodes = station2_action[b, stage2_tasks]
                
                station1_coord = nodes["coord"][b, stage2_station1_nodes]
                station2_coord = nodes["coord"][b, stage2_station2_nodes]
                
                station1_embd = self.station1_drone_proj_in(
                    self.STATION1_DRONE[None, :].expand(n_s2, -1) +
                    requests_feature[b, stage2_tasks] +
                    self.coord_proj_in(station1_coord)
                )
                station2_embd = self.station2_drone_proj_in(
                    self.STATION2_DRONE[None, :].expand(n_s2, -1) +
                    requests_feature[b, stage2_tasks] +
                    self.coord_proj_in(station2_coord)
                )
            else:
                station1_embd = torch.empty([0, self.n_embd], device=device)
                station2_embd = torch.empty([0, self.n_embd], device=device)
            
            drone_nodes = torch.cat([
                depot_drone_embd + global_feature[b, None, :],
                station1_embd + global_feature[b, None, :],
                station2_embd + global_feature[b, None, :]
            ], dim=0)
            
            n_drone_nodes = n_drone + n_s2 * 2
            drone_nodes_mask = torch.ones(n_drone_nodes, dtype=torch.bool, device=device)
            
            drone_nodes_embd_list.append(drone_nodes)
            drone_nodes_mask_list.append(drone_nodes_mask)
        
        max_drone_nodes = max([e.shape[0] for e in drone_nodes_embd_list] + [n_drone])
        drone_nodes_embd_padded = torch.zeros(
            batch_size, max_drone_nodes, self.n_embd, device=device
        )
        drone_nodes_mask_padded = torch.zeros(
            batch_size, max_drone_nodes, dtype=torch.bool, device=device
        )
        
        for b in range(batch_size):
            n_nodes = drone_nodes_embd_list[b].shape[0]
            drone_nodes_embd_padded[b, :n_nodes] = drone_nodes_embd_list[b]
            drone_nodes_mask_padded[b, :n_nodes] = drone_nodes_mask_list[b]
        
        drone_nodes_embd = self.encoder_drone.forward(
            drone_nodes_embd_padded, drone_nodes_mask_padded, None
        )
        
        # Global embedding（简化版本）
        global_embd = global_feature  # 简化处理
        
        values = self.value_linear(global_embd).squeeze(-1)
        if only_critic:
            return values

        # ========== Courier任务选择（统一处理Stage1和Stage3）==========
        courier_task_target = couriers["task_target"]  # [B, n_courier]
        
        # 获取当前任务的embedding（需要根据任务索引映射到节点）
        courier_target_task_embd = torch.zeros(
            batch_size, n_courier, self.n_embd, device=device
        )
        for b in range(batch_size):
            for c in range(n_courier):
                task_idx = courier_task_target[b, c]
                if task_idx >= 0 and task_idx < courier_task_map[b].numel():
                    request_idx = courier_task_map[b][task_idx]
                    if request_idx >= 0:
                        # 根据任务类型找到对应的节点位置
                        task_type = courier_task_type[b][task_idx]
                        if task_type == 1:  # Stage1任务
                            # 节点位置：depot(0), from(n_courier), station1(n_courier+n_s1)
                            node_idx = n_courier + task_idx  # from节点
                        elif task_type == 3:  # Stage3任务
                            # 节点位置：depot(0), from(n_courier), station1(n_courier+n_s1), 
                            # station2(n_courier+n_s1*2), to(n_courier+n_s1*2+n_s3)
                            n_s1 = n_stage1_tasks_list[b]
                            node_idx = n_courier + n_s1 * 2 + (task_idx - n_s1)  # station2节点
                        else:
                            node_idx = 0  # depot
                        
                        if node_idx < courier_nodes_embd[b].shape[0]:
                            courier_target_task_embd[b, c] = courier_nodes_embd[b, node_idx]
                        else:
                            courier_target_task_embd[b, c] = self.NO_TARGET
                    else:
                        courier_target_task_embd[b, c] = self.NO_TARGET
                else:
                    courier_target_task_embd[b, c] = self.NO_TARGET
        
        # Courier决策
        comm_courier = torch.cat((couriers["space"], courier_target_task_embd.flatten(1)), -1)
        h_courier = torch.cat((
            courier_target_task_embd,
            couriers["space"][:, :, None],
            global_embd[:, None, :].expand(-1, n_courier, -1),
            comm_courier[:, None, :].expand(-1, n_courier, -1)
        ), -1)
        
        # 构建任务key（从任务节点embedding中提取）
        # 简化：使用所有任务的embedding作为key
        courier_task_keys = []
        for b in range(batch_size):
            n_tasks = courier_task_map[b].numel()
            task_keys = []
            for t in range(n_tasks):
                request_idx = courier_task_map[b][t]
                if request_idx >= 0:
                    task_type = courier_task_type[b][t]
                    if task_type == 1:  # Stage1: 使用station1节点
                        node_idx = n_courier + n_stage1_tasks_list[b] + t
                    elif task_type == 3:  # Stage3: 使用to节点
                        n_s1 = n_stage1_tasks_list[b]
                        node_idx = n_courier + n_s1 * 2 + n_s1 + (t - n_s1)
                    else:
                        node_idx = 0
                    
                    if node_idx < courier_nodes_embd[b].shape[0]:
                        task_keys.append(courier_nodes_embd[b, node_idx])
                    else:
                        task_keys.append(torch.zeros(self.n_embd, device=device))
                else:
                    task_keys.append(torch.zeros(self.n_embd, device=device))
            
            if len(task_keys) > 0:
                task_keys_tensor = torch.stack(task_keys, dim=0)  # [n_tasks, n_embd]
            else:
                task_keys_tensor = torch.zeros([0, self.n_embd], device=device)
            courier_task_keys.append(task_keys_tensor)
        
        # Padding
        max_courier_tasks = max([k.shape[0] for k in courier_task_keys] + [1])
        courier_task_keys_padded = torch.zeros(
            batch_size, max_courier_tasks, self.n_embd, device=device
        )
        courier_task_keys_mask = torch.zeros(
            batch_size, max_courier_tasks, dtype=torch.bool, device=device
        )
        
        for b in range(batch_size):
            n_tasks = courier_task_keys[b].shape[0]
            if n_tasks > 0:
                courier_task_keys_padded[b, :n_tasks] = courier_task_keys[b]
                courier_task_keys_mask[b, :n_tasks] = True
        
        g_courier = self.cross_attn.forward(
            self.proj_h(h_courier), courier_nodes_embd, courier_nodes_embd
        )
        q_courier = self.to_q_g(g_courier)  # [B, n_courier, n_embd]
        k_courier = self.to_k_g(courier_task_keys_padded)  # [B, max_tasks, n_embd]
        
        D = 10
        d = self.n_embd / self.n_head
        k_courier_with_noaction = torch.cat((
            k_courier, self.NO_ACTION.expand(batch_size, 1, -1)
        ), 1)
        u_courier = D * (q_courier @ k_courier_with_noaction.transpose(-1, -2) / d ** 0.5).tanh()
        
        # Courier mask
        cur_courier_request = courier_request.clone()
        cur_courier_space = couriers["space"].clone()
        
        # 构建mask
        courier_mask = torch.zeros(
            batch_size, n_courier, max_courier_tasks + 1, device=device, dtype=torch.bool
        )
        
        for b in range(batch_size):
            n_tasks = courier_task_map[b].numel()
            n_s1 = n_stage1_tasks_list[b]
            
            for c in range(n_courier):
                if not courier_decode_mask[b, c]:
                    continue
                
                # Stage3任务mask（优先）
                for t in range(n_s1, n_tasks):
                    request_idx = courier_task_map[b][t]
                    if request_idx < 0:
                        continue
                    # 检查状态：drone_request == 4 且 courier_request != 5
                    if (drone_request[b, :, request_idx] == 4).any() and \
                       (courier_request[b, :, request_idx] != 5).all() and \
                       cur_courier_space[b, c] >= requests["volumn"][b, request_idx] and \
                       couriers["target"][b, c] == station2_action[b, request_idx]:
                        courier_mask[b, c, t] = True
                
                # Stage1任务mask（如果没有Stage3任务可选）
                has_stage3 = courier_mask[b, c, n_s1:n_tasks].any()
                if not has_stage3:
                    for t in range(n_s1):
                        request_idx = courier_task_map[b][t]
                        if request_idx < 0:
                            continue
                        # 检查状态：courier_request == 0
                        if (courier_request[b, :, request_idx] == 0).all() and \
                           cur_courier_space[b, c] >= requests["volumn"][b, request_idx] and \
                           couriers["target"][b, c] == requests["from"][b, request_idx]:
                            courier_mask[b, c, t] = True
                
                # 如果没有任务可选，选择NO_ACTION
                if not courier_mask[b, c, :max_courier_tasks].any():
                    courier_mask[b, c, max_courier_tasks] = True
        
        u_courier[:, :, :-1].masked_fill_(~courier_mask[:, :, :-1], -torch.inf)
        u_courier[:, :, -1].masked_fill_(
            (u_courier[:, :, :-1] == -torch.inf).all(-1), 1
        )
        p_courier = F.softmax(u_courier, -1)
        p_courier = p_courier.masked_fill(
            torch.cat((
                torch.zeros_like(p_courier, dtype=torch.bool)[..., :-1],
                (p_courier[..., :-1].sum(-1) != 0).unsqueeze(-1)
            ), dim=-1), 0
        )
        
        cate_courier = Categorical(probs=p_courier)
        
        # ========== Drone任务选择（Stage2）==========
        drone_task_target = drones["task_target"]
        
        # 类似处理drone
        drone_target_task_embd = torch.zeros(
            batch_size, n_drone, self.n_embd, device=device
        )
        for b in range(batch_size):
            for d in range(n_drone):
                task_idx = drone_task_target[b, d]
                if task_idx >= 0 and task_idx < drone_task_map[b].numel():
                    request_idx = drone_task_map[b][task_idx]
                    if request_idx >= 0:
                        # Stage2任务：使用station1节点
                        node_idx = n_drone + task_idx
                        if node_idx < drone_nodes_embd[b].shape[0]:
                            drone_target_task_embd[b, d] = drone_nodes_embd[b, node_idx]
                        else:
                            drone_target_task_embd[b, d] = self.NO_TARGET
                    else:
                        drone_target_task_embd[b, d] = self.NO_TARGET
                else:
                    drone_target_task_embd[b, d] = self.NO_TARGET
        
        comm_drone = torch.cat((drones["space"], drone_target_task_embd.flatten(1)), -1)
        h_drone = torch.cat((
            drone_target_task_embd,
            drones["space"][:, :, None],
            global_embd[:, None, :].expand(-1, n_drone, -1),
            comm_drone[:, None, :].expand(-1, n_drone, -1)
        ), -1)
        
        drone_task_keys = []
        for b in range(batch_size):
            n_tasks = drone_task_map[b].numel()
            task_keys = []
            for t in range(n_tasks):
                request_idx = drone_task_map[b][t]
                if request_idx >= 0:
                    # Stage2: 使用station2节点
                    node_idx = n_drone + n_tasks + t
                    if node_idx < drone_nodes_embd[b].shape[0]:
                        task_keys.append(drone_nodes_embd[b, node_idx])
                    else:
                        task_keys.append(torch.zeros(self.n_embd, device=device))
                else:
                    task_keys.append(torch.zeros(self.n_embd, device=device))
            
            if len(task_keys) > 0:
                task_keys_tensor = torch.stack(task_keys, dim=0)
            else:
                task_keys_tensor = torch.zeros([0, self.n_embd], device=device)
            drone_task_keys.append(task_keys_tensor)
        
        max_drone_tasks = max([k.shape[0] for k in drone_task_keys] + [1])
        drone_task_keys_padded = torch.zeros(
            batch_size, max_drone_tasks, self.n_embd, device=device
        )
        drone_task_keys_mask = torch.zeros(
            batch_size, max_drone_tasks, dtype=torch.bool, device=device
        )
        
        for b in range(batch_size):
            n_tasks = drone_task_keys[b].shape[0]
            if n_tasks > 0:
                drone_task_keys_padded[b, :n_tasks] = drone_task_keys[b]
                drone_task_keys_mask[b, :n_tasks] = True
        
        g_drone = self.cross_attn_drone.forward(
            self.proj_h_drone(h_drone), drone_nodes_embd, drone_nodes_embd
        )
        q_drone = self.to_q_g_drone(g_drone)
        k_drone = self.to_k_g_drone(drone_task_keys_padded)
        
        k_drone_with_noaction = torch.cat((
            k_drone, self.NO_ACTION_DRONE.expand(batch_size, 1, -1)
        ), 1)
        u_drone = D * (q_drone @ k_drone_with_noaction.transpose(-1, -2) / d ** 0.5).tanh()
        
        # Drone mask
        cur_drone_request = drone_request.clone()
        drone_mask = torch.zeros(
            batch_size, n_drone, max_drone_tasks + 1, device=device, dtype=torch.bool
        )
        
        for b in range(batch_size):
            n_tasks = drone_task_map[b].numel()
            for d in range(n_drone):
                if not drone_decode_mask[b, d]:
                    continue
                
                for t in range(n_tasks):
                    request_idx = drone_task_map[b][t]
                    if request_idx < 0:
                        continue
                    # 检查状态：courier_request == 3 且 drone_request == 0
                    if (courier_request[b, :, request_idx] == 3).any() and \
                       (drone_request[b, :, request_idx] == 0).all() and \
                       drones["space"][b, d] >= requests["volumn"][b, request_idx] and \
                       drones["target"][b, d] == station1_action[b, request_idx]:
                        drone_mask[b, d, t] = True
                
                if not drone_mask[b, d, :max_drone_tasks].any():
                    drone_mask[b, d, max_drone_tasks] = True
        
        u_drone[:, :, :-1].masked_fill_(~drone_mask[:, :, :-1], -torch.inf)
        u_drone[:, :, -1].masked_fill_(
            (u_drone[:, :, :-1] == -torch.inf).all(-1), 1
        )
        p_drone = F.softmax(u_drone, -1)
        p_drone = p_drone.masked_fill(
            torch.cat((
                torch.zeros_like(p_drone, dtype=torch.bool)[..., :-1],
                (p_drone[..., :-1].sum(-1) != 0).unsqueeze(-1)
            ), dim=-1), 0
        )
        
        cate_drone = Categorical(probs=p_drone)
        
        # ========== 采样和动作解码 ==========
        is_decoding = input_actions is None
        if is_decoding:
            sampled_courier_task = cate_courier.sample()  # [B, n_courier]
            sampled_drone_task = cate_drone.sample()  # [B, n_drone]
            
            # 冲突处理
            new_courier_task = sampled_courier_task.clone()
            new_drone_task = sampled_drone_task.clone()
            
            # Courier冲突处理
            for c in range(n_courier):
                courier_mask_c = courier_decode_mask[:, c] & (new_courier_task[:, c] < max_courier_tasks)
                courier_batch = batch_arange[courier_mask_c]
                for b_idx in courier_batch:
                    task_idx = new_courier_task[b_idx, c]
                    if task_idx >= 0 and task_idx < courier_task_map[b_idx].numel():
                        request_idx = courier_task_map[b_idx][task_idx]
                        if request_idx >= 0:
                            # 检查冲突
                            conflict = False
                            for c2 in range(c):
                                if courier_decode_mask[b_idx, c2] and \
                                   new_courier_task[b_idx, c2] == task_idx:
                                    conflict = True
                                    break
                            if conflict:
                                new_courier_task[b_idx, c] = max_courier_tasks  # NO_ACTION
            
            # Drone冲突处理
            for d in range(n_drone):
                drone_mask_d = drone_decode_mask[:, d] & (new_drone_task[:, d] < max_drone_tasks)
                drone_batch = batch_arange[drone_mask_d]
                for b_idx in drone_batch:
                    task_idx = new_drone_task[b_idx, d]
                    if task_idx >= 0 and task_idx < drone_task_map[b_idx].numel():
                        request_idx = drone_task_map[b_idx][task_idx]
                        if request_idx >= 0:
                            # 检查冲突
                            conflict = False
                            for d2 in range(d):
                                if drone_decode_mask[b_idx, d2] and \
                                   new_drone_task[b_idx, d2] == task_idx:
                                    conflict = True
                                    break
                            if conflict:
                                new_drone_task[b_idx, d] = max_drone_tasks  # NO_ACTION
            
            # 解码动作
            couriers_action = couriers["target"].clone()
            drones_action = drones["target"].clone()
            requests_couriers_action = torch.full(
                [batch_size, n_request], n_courier, dtype=torch.int64, device=device
            )
            requests_drones_action = torch.full(
                [batch_size, n_request], n_drone, dtype=torch.int64, device=device
            )
            
            # Courier动作解码
            for b in range(batch_size):
                for c in range(n_courier):
                    if courier_decode_mask[b, c]:
                        task_idx = new_courier_task[b, c]
                        if task_idx < max_courier_tasks and task_idx < courier_task_map[b].numel():
                            request_idx = courier_task_map[b][task_idx]
                            if request_idx >= 0:
                                task_type = courier_task_type[b][task_idx]
                                if task_type == 1:  # Stage1
                                    couriers_action[b, c] = requests["from"][b, request_idx]
                                    requests_couriers_action[b, request_idx] = c
                                elif task_type == 3:  # Stage3
                                    couriers_action[b, c] = station2_action[b, request_idx]
                                    requests_couriers_action[b, request_idx] = c
            
            # Drone动作解码
            for b in range(batch_size):
                for d in range(n_drone):
                    if drone_decode_mask[b, d]:
                        task_idx = new_drone_task[b, d]
                        if task_idx < max_drone_tasks and task_idx < drone_task_map[b].numel():
                            request_idx = drone_task_map[b][task_idx]
                            if request_idx >= 0:
                                drones_action[b, d] = station1_action[b, request_idx]
                                requests_drones_action[b, request_idx] = d
            
            courier_new_task_target = new_courier_task
            drone_new_task_target = new_drone_task
        else:
            courier_new_task_target = input_actions["courier_new_task_target"]
            drone_new_task_target = input_actions["drone_new_task_target"]
            couriers_action = input_actions["courier"]
            drones_action = input_actions["drone"]
            requests_couriers_action = input_actions["request_courier"]
            requests_drones_action = input_actions["request_drone"]
            station1_action = input_actions.get("station1", station1_action)
            station2_action = input_actions.get("station2", station2_action)
            sampled_courier_task = input_actions.get("sampled_courier_task", courier_new_task_target)
            sampled_drone_task = input_actions.get("sampled_drone_task", drone_new_task_target)
        
        log_prob = cate_courier.log_prob(sampled_courier_task).sum(-1) + \
                   cate_drone.log_prob(sampled_drone_task).sum(-1)
        entropy = cate_courier.entropy().sum(-1) + cate_drone.entropy().sum(-1)
        
        return {
            "station1": station1_action,
            "station2": station2_action,
            "courier": couriers_action,
            "drone": drones_action,
            "request_courier": requests_couriers_action,
            "request_drone": requests_drones_action,
            "courier_new_task_target": courier_new_task_target,
            "drone_new_task_target": drone_new_task_target,
            "sampled_courier_task": sampled_courier_task,
            "sampled_drone_task": sampled_drone_task
        }, log_prob, entropy, values

MAPDP.register_algo_specific()