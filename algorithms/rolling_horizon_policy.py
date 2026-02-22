import multiprocessing as mp
import itertools

import torch
from .cpsat import CPSAT
from .nearest import NearestHeuristic
from .genetic import GeneticAlgorithm
from .simulated_annealing import SimulatedAnnealing

def call_solve(solver):
    return solver.solve(show=False)

def assign_stations(_global, requests, batch_size, n_request, n_node, device):
    station1_action = torch.full([batch_size, n_request], n_node, dtype=torch.int64, device=device)
    station2_action = torch.full([batch_size, n_request], n_node, dtype=torch.int64, device=device)
    
    # 取最近的station作为station1和station2
    dist_matrix = _global["node_node"].to(torch.float32)
    for b in range(batch_size):
        stations = _global["station_idx"][b,:-1]
        request_from = requests["from"][b]
        request_to = requests["to"][b]
        
        from_dists = dist_matrix[b, request_from[:, None], stations[None, :]]
        min_station1_idx = from_dists.argmin(dim=1)
        station1_action[b] = stations[min_station1_idx]
        
        to_dists = dist_matrix[b, request_to[:, None], stations[None, :]]

        to_dists_excluded = to_dists.clone()
        exclude_mask = torch.zeros_like(to_dists_excluded, dtype=torch.bool)
        exclude_mask[torch.arange(len(min_station1_idx)), min_station1_idx] = True

        to_dists_excluded = to_dists_excluded.masked_fill(exclude_mask, float('inf'))

        min_station2_idx = to_dists_excluded.argmin(dim=1)
        station2_action[b] = stations[min_station2_idx]

        assert (station1_action[b] != station2_action[b]).all(), "Station assignment conflict not resolved"
    
    return station1_action, station2_action

class RollingHorizonPolicy:
    def __init__(self, args, use_hindsight=False):
        self.env_args = args.env_args
        self.algorithm = args.algorithm
        self.eval_episodes = args.eval_episodes
        
        self.solver_cls = {
            "cpsat": CPSAT,
            "nearest": NearestHeuristic,
            "ga": GeneticAlgorithm,
            "sa": SimulatedAnnealing,
        }[args.algorithm]

        # Tunable Hyper Parameter
        self.use_hindsight = use_hindsight
        if not use_hindsight:
            self.max_consider_frame = self.env_args["max_dist"] # changed from 3 to 2 # TODO
            self.reschedule_T = self.env_args["max_dist"]  # 每隔 reschedule_T 帧重新规划一次
            assert self.reschedule_T <= self.max_consider_frame
            # self.max_consider_frame = self.env_args["n_frame"]
            # self.reschedule_T = self.env_args["n_frame"]
        else:
            self.max_consider_frame = 114514
            self.reschedule_T = 114514

        if self.env_args["debug"] or args.eval_episodes == 1:
            self.map = map
        else:
            num_cores = mp.cpu_count()
            self.pool = mp.Pool(min(args.eval_episodes, num_cores))
            self.map = self.pool.map
        # cached infos
        self.solutions = None
        self.cached_extractor = None
        self.requests_mask = None
    
    def update(self, _global, nodes, couriers, drones, requests, extractor):
        device = requests["value"].device
        batch_size = _global["n_exist_requests"].shape[0]

        n_node = _global["node_node"].shape[-1]
        n_courier = couriers["capacity"].shape[1]
        n_drone = drones["target"].shape[1]
        
        n_request = requests["value"].shape[1] # max_consider_requests
        assert (_global["frame"] == _global["frame"][0]).all()
        frame = _global["frame"][0].item()
        requests_mask = torch.arange(n_request, device=device) < _global["n_consider_requests"][:, None]
        solvers = []
        for i in range(batch_size):
            frame_left = min(self.env_args["n_frame"] - frame, self.max_consider_frame) # large frame_left 搜索不动
            is_courier_delivering = (_global["courier_request"][i, :, requests_mask[i]] == 2)
            pre_load_courier_stage1 = is_courier_delivering & ((_global["courier_request"][i, :, requests_mask[i]] != 3).all(0, keepdim=True)) 
            pre_load_courier_stage3 = is_courier_delivering & ((_global["drone_request"][i, :, requests_mask[i]] == 4).any(0, keepdim=True))
            pre_load_drone = _global["drone_request"][i, :, requests_mask[i]] == 2
            # 到达阶段2 或 阶段3 但是还没有载具来运输
            wait_stage2 = (_global["courier_request"][i, :, requests_mask[i]] == 3).any(0) & (_global["drone_request"][i, :, requests_mask[i]] < 2).all(0) 
            wait_stage3 = (_global["drone_request"][i, :, requests_mask[i]] == 4).any(0) & (_global["courier_request"][i, :, requests_mask[i]] != 2).all(0) 
            # # 使用 drone 的状态来区分 courier 的 stage1 和 stage3
            # # 如果 drone 已经完成配送 (状态 4)，那么 courier 的配送 (状态 2) 一定是 stage3
            # is_drone_done = (_global["drone_request"][i, :, requests_mask[i]] == 4).any(0, keepdim=True)
            # is_courier_delivering = (_global["courier_request"][i, :, requests_mask[i]] == 2)
            
            # pre_load_courier_stage3 = is_courier_delivering & is_drone_done
            # pre_load_courier_stage1 = is_courier_delivering & (~is_drone_done)
            
            # pre_load_drone = _global["drone_request"][i, :, requests_mask[i]] == 2

            
            appear = (requests["appear"][i, requests_mask[i]] - frame).clamp_min(0).masked_fill(pre_load_courier_stage1.any(dim=0) | pre_load_courier_stage3.any(dim=0) | pre_load_drone.any(dim=0), -1)
            objective_scale = 1. if self.env_args["dist_cost_courier"] == 0. else self.env_args["dist_cost_courier"]
            
            # 防止stage1和3使用同一个courier
            mask = (_global["courier_request"][i, :, requests_mask[i]] == 3)
            
            has_any = mask.any(dim=0)
            first_idx = mask.int().argmax(dim=0)
            courier_stage1_temp = torch.where(has_any, first_idx, torch.full_like(first_idx, n_courier))
            
            solver = self.solver_cls(
                n_node, 
                _global["n_consider_requests"][i].item(), 
                n_courier,
                n_drone, 
                frame_left,
                couriers["target"][i].tolist(),
                drones["target"][i].tolist(), 
                couriers["capacity"][i].tolist(), 
                couriers["time_left"][i].clamp_max(frame_left).tolist(),
                drones["time_left"][i].clamp_max(frame_left).tolist(),
                _global["node_node"][i].round().to(torch.int64).tolist(),
                # TODO 如果courier和drone的cost不同需要修改 
                (_global["node_node"][i] * self.env_args["dist_cost_courier"] / objective_scale).round().to(torch.int64).tolist(),
                (_global["node_node"][i] * self.env_args["dist_cost_drone"] / objective_scale).round().to(torch.int64).tolist(),
                requests["from"][i][requests_mask[i]].tolist(), 
                requests["to"][i][requests_mask[i]].tolist(),
                requests["station1"][i][requests_mask[i]].tolist(),
                requests["station2"][i][requests_mask[i]].tolist(),
                appear.tolist(),
                value=(requests["value"][i][requests_mask[i]] / objective_scale).round().to(torch.int64).tolist(),
                penalty=torch.zeros_like(requests["from"][i][requests_mask[i]]).tolist(),
                pre_load_K_stage1=pre_load_courier_stage1.tolist(),
                pre_load_K_stage3=pre_load_courier_stage3.tolist(),
                pre_load_D=pre_load_drone.tolist(),
                wait_stage2 = wait_stage2.tolist(),
                wait_stage3 = wait_stage3.tolist(),
                objective_scale = objective_scale,
                env_args=self.env_args,
                courier_stage1_temp = courier_stage1_temp.tolist(),
                drone_speed_ratio = self.env_args.get('drone_speed_ratio', 4),
            )
            solvers.append(solver)

        num_cores = mp.cpu_count()
        # num_cores = 4
        if num_cores < self.eval_episodes:
            batch_size = num_cores
            num_batches = (self.eval_episodes + batch_size - 1) // batch_size
            self.pool = mp.Pool(batch_size)
            
            def batched_map(func, iterable):
                results = []
                for i in range(num_batches):
                    batch = list(itertools.islice(iterable, i * batch_size, (i + 1) * batch_size))
                    batch_results = self.pool.map(func, batch)
                    results.extend(batch_results)
                return results
            self.solutions = list(batched_map(call_solve, solvers))
        else:
            self.solutions = list(self.map(call_solve, solvers))
        self.cached_extractor = extractor
        self.requests_mask = requests_mask
    
    def hindsight_update(self, _global, nodes, couriers, drones, requests, global_n_consider_requests):
        device = requests["value"].device
        batch_size = requests["value"].shape[0]
        n_request = requests["value"].shape[1] # max_consider_requests
        n_node = _global["node_node"].shape[-1]
        extractor = torch.arange(n_request, device=device).repeat(batch_size, 1)

        station1_action, station2_action = assign_stations(_global, requests, batch_size, n_request, n_node, device)
        requests["station1"] = station1_action.clone()
        requests["station2"] = station2_action.clone()

        _global["n_consider_requests"] = global_n_consider_requests
        _global["frame"] = torch.full((batch_size, ), _global["frame"], device=device)
        self.update(_global, nodes, couriers, drones, requests, extractor)
        
    def act(self, obs, rnn_states_actor, deterministic=True):
        assert deterministic is True
        
        nodes = obs["nodes"]
        couriers = obs["couriers"]
        drones = obs["drones"]
        
        requests = obs["requests"]
        _global = obs["global"]
        extractor = obs["extractor"]
        device = nodes["n_courier"].device

        batch_size = _global["n_exist_requests"].shape[0]
        n_courier = couriers["target"].shape[1]
        n_drone = drones["target"].shape[1]

        n_node = nodes["n_courier"].shape[1]
        
        n_request = requests["value"].shape[1] # max_consider_requests
        assert (_global["frame"] == _global["frame"][0]).all()
        frame = _global["frame"][0].item()
        print(frame)
        # assert (requests["volumn"] == 1).all()
        
        courier_arange = torch.arange(n_courier, device=device)
        drone_arange = torch.arange(n_drone, device=device)
        
        couriers_action = couriers["target"].clone()
        drones_action = drones["target"].clone()
        
        requests_couriers_action = torch.full([batch_size, n_request], n_courier, dtype=torch.int64, device=device)
        requests_drones_action = torch.full([batch_size, n_request], n_drone, dtype=torch.int64, device=device)
        
        # 取最近的station作为station1和station2
        station1_action, station2_action = assign_stations(_global, requests, batch_size, n_request, n_node, device)
        requests["station1"] = station1_action.clone()
        requests["station2"] = station2_action.clone()

        offway_mask_courier = couriers["time_left"] == 0
        offway_mask_drone = drones["time_left"] == 0
        if (frame == 0 and not self.use_hindsight) or (frame !=0 and frame % self.reschedule_T == 0):
            self.update(_global, nodes, couriers, drones, requests, extractor)

        def _assign(actions, assign_veh, assign_req_problem_id, i):
            if assign_veh.numel() == 0:
                return
            cached_id = torch.where(self.requests_mask[i])[0][assign_req_problem_id]
            global_id = self.cached_extractor[i, cached_id]
            match = global_id[:, None] == extractor[i][None, :]
            idx = match.to(torch.int64).argmax(-1)
            if self.use_hindsight:
                valid = match.any(-1)
                actions[i, idx[valid]] = assign_veh[valid]
            else:
                assert (extractor[i, idx] == global_id).all()
                actions[i, idx] = assign_veh

        for i, (pickup1, pickup2, pickup_d, location_k, location_d) in enumerate(self.solutions):
            pickup1 = torch.tensor(pickup1, device=device)[..., frame % self.reschedule_T:]
            pickup2 = torch.tensor(pickup2, device=device)[..., frame % self.reschedule_T:]
            location_k = torch.tensor(location_k, device=device)[..., frame % self.reschedule_T:]
            pickup_d = torch.tensor(pickup_d, device=device)[..., frame % self.reschedule_T:]
            location_d = torch.tensor(location_d, device=device)[..., frame % self.reschedule_T:]
            assert (pickup1[..., 0].sum(0) <= 1).all(), "stage1 每个物品只能分配给 1 个courier"
            assert (pickup2[..., 0].sum(0) <= 1).all(), "stage3 每个物品只能分配给 1 个courier"
            assert (pickup_d[..., 0].sum(0) <= 1).all(), "stage2 每个物品只能分配给 1 个drone"
            assign_courier1, assign_requests_problem_id1 = torch.where(pickup1[..., 0])
            assign_courier2, assign_requests_problem_id2 = torch.where(pickup2[..., 0])
            assign_drone, assign_requests_problem_id_d = torch.where(pickup_d[..., 0])

            _assign(requests_couriers_action, assign_courier1, assign_requests_problem_id1, i)
            _assign(requests_couriers_action, assign_courier2, assign_requests_problem_id2, i)
            _assign(requests_drones_action, assign_drone, assign_requests_problem_id_d, i)
            
            next_hop_t = (location_k[:, 1:] != -1).to(torch.int64).argmax(-1) + 1
            action_courier_mask = offway_mask_courier[i] & (location_k[:, 0] == couriers["target"][i]) & (location_k[courier_arange, next_hop_t] != -1)
            couriers_action[i, action_courier_mask] = location_k[courier_arange[action_courier_mask], next_hop_t[action_courier_mask]]
            
            next_hop_t = (location_d[:, 1:] != -1).to(torch.int64).argmax(-1) + 1
            action_drone_mask = offway_mask_drone[i] & (location_d[:, 0] == drones["target"][i]) & (location_d[drone_arange, next_hop_t] != -1)
            drones_action[i, action_drone_mask] = location_d[drone_arange[action_drone_mask], next_hop_t[action_drone_mask]]
        # 避免给已有无人机关联的请求再次分配无人机
        has_drone_relation = (_global["drone_request"] != 0).any(dim=1)
        requests_drones_action = requests_drones_action.masked_fill(has_drone_relation, n_drone)
        # print(requests_drones_action[0,7])
        # import pdb
        # pdb.set_trace()
        return {"station1": station1_action, "station2": station2_action, "request_courier": requests_couriers_action, "request_drone": requests_drones_action, "courier": couriers_action, "drone": drones_action}, rnn_states_actor
    
    def eval(self):
        pass