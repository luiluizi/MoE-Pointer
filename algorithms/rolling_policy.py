import multiprocessing as mp

import torch
# from .cpsat import CPSAT
from .nearest import NearestHeuristic
from .genetic import GeneticAlgorithm
from .annealing import SimulatedAnnealing

def call_solve(solver):
    return solver.solve(show=False)

class RollingHorizonPolicy:
    def __init__(self, args, use_hindsight=False):
        self.env_args = args.env_args

        self.solver_cls = {
            # "cpsat": CPSAT,
            "nearest": NearestHeuristic,
            "ga": GeneticAlgorithm,
            "sa": SimulatedAnnealing,
        }[args.algorithm]

        # Tunable Hyper Parameter
        self.use_hindsight = use_hindsight
        if not use_hindsight:
            self.max_consider_frame = self.env_args["max_dist"] * 2 # changed from 3 to 2 # TODO
            self.reschedule_T = self.env_args["max_dist"] # 每隔 reschedule_T 帧重新规划一次
            assert self.reschedule_T <= self.max_consider_frame
            # self.max_consider_frame = self.env_args["n_frame"]
            # self.reschedule_T = self.env_args["n_frame"]
        else:
            self.max_consider_frame = 114514
            self.reschedule_T = 114514

        if self.env_args["debug"] or args.eval_episodes == 1:
            self.map = map
        else:
            self.pool = mp.Pool(min(args.eval_episodes, 64))
            self.map = self.pool.map
        

        # cached infos
        self.solutions = None
        self.cached_extractor = None
        self.requests_mask = None
    
    def update(self, _global, nodes, vehicles, requests, extractor):
        device = requests["value"].device
        batch_size = _global["n_exist_requests"].shape[0]

        n_node = _global["node_node"].shape[-1]
        n_vehicle = vehicles["capacity"].shape[1]
        n_request = requests["value"].shape[1] # max_consider_requests
        assert (_global["frame"] == _global["frame"][0]).all()
        frame = _global["frame"][0].item()

        requests_mask = torch.arange(n_request, device=device) < _global["n_consider_requests"][:, None]

        solvers = []
        for i in range(batch_size):
            # 大 frame_left -> 车辆懒惰
            # 小 frame_left -> 车辆投机
            frame_left = min(self.env_args["n_frame"] - frame, self.max_consider_frame) # large frame_left 搜索不动
            pre_load = _global["vehicle_request"][i, :, requests_mask[i]] == 2
            appear = (requests["appear"][i, requests_mask[i]] - frame).clamp_min(0).masked_fill(pre_load.any(dim=0), -1)
            objective_scale = 1. if self.env_args["dist_cost"] == 0. else self.env_args["dist_cost"]
            solver = self.solver_cls(
                n_node, _global["n_consider_requests"][i].item(), n_vehicle, frame_left,
                vehicles["target"][i].tolist(), vehicles["capacity"][i].tolist(), vehicles["time_left"][i].clamp_max(frame_left).tolist(),
                _global["node_node"][i].round().to(torch.int64).tolist(), (_global["node_node"][i] * self.env_args["dist_cost"] / objective_scale).round().to(torch.int64).tolist(),
                requests["from"][i][requests_mask[i]].tolist(), requests["to"][i][requests_mask[i]].tolist(),
                appear.tolist(),
                (requests["value"][i][requests_mask[i]] / objective_scale).round().to(torch.int64).tolist(),
                penalty=torch.zeros_like(requests["from"][i][requests_mask[i]]).tolist(),
                pre_load=pre_load.tolist(),
                objective_scale = objective_scale,
                env_args=self.env_args
            )
            solvers.append(solver)
        self.solutions = list(self.map(call_solve, solvers))
        self.cached_extractor = extractor
        self.requests_mask = requests_mask
    
    def hindsight_update(self, _global, nodes, vehicles, requests, global_n_consider_requests):
        device = requests["value"].device
        batch_size = requests["value"].shape[0]
        n_request = requests["value"].shape[1] # max_consider_requests

        extractor = torch.arange(n_request, device=device).repeat(batch_size, 1)

        _global["n_consider_requests"] = global_n_consider_requests
        _global["frame"] = torch.full((batch_size, ), _global["frame"], device=device)
        self.update(_global, nodes, vehicles, requests, extractor)
        
    def act(self, obs, rnn_states_actor, deterministic=True):
        assert deterministic is True

        nodes = obs["nodes"]
        vehicles = obs["vehicles"]
        requests = obs["requests"]
        _global = obs["global"]
        extractor = obs["extractor"]
        device = nodes["n_vehicle"].device

        batch_size = _global["n_exist_requests"].shape[0]
        n_vehicle = vehicles["capacity"].shape[1]
        n_request = requests["value"].shape[1] # max_consider_requests
        assert (_global["frame"] == _global["frame"][0]).all()
        frame = _global["frame"][0].item()

        assert (requests["volumn"] == 1).all()
        
        vehicle_arange = torch.arange(n_vehicle, device=device)

        # vehicles_action = torch.full([batch_size, n_vehicle], 114514, dtype=torch.int64, device=device)
        vehicles_action = vehicles["target"].clone()
        requests_action = torch.full([batch_size, n_request], n_vehicle, dtype=torch.int64, device=device)

        offway_mask = vehicles["time_left"] == 0

        if (frame == 0 and not self.use_hindsight) or (frame !=0 and frame % self.reschedule_T == 0):
            self.update(_global, nodes, vehicles, requests, extractor)
        
        for i, (pickup, location) in enumerate(self.solutions):
            pickup = torch.tensor(pickup, device=device)[..., frame % self.reschedule_T:]
            location = torch.tensor(location, device=device)[..., frame % self.reschedule_T:]
            
            assert (pickup[..., 0].sum(0) <= 1).all(), "每个物品只能分配给 1 辆车"
            assign_vehicles, assign_requests_problem_id = torch.where(pickup[..., 0])
            assign_requests_cached_id = torch.where(self.requests_mask[i])[0][assign_requests_problem_id]
            assign_requests_global_id = self.cached_extractor[i, assign_requests_cached_id]
            assign_requests_cur_id = (assign_requests_global_id[..., None] == extractor[i]).to(torch.int64).argmax(-1)
            assert (extractor[i, assign_requests_cur_id] == assign_requests_global_id).all()
            requests_action[i, assign_requests_cur_id] = assign_vehicles

            next_hop_t = (location[:, 1:] != -1).to(torch.int64).argmax(-1) + 1
            action_vehicle_mask = offway_mask[i] & (location[:, 0] == vehicles["target"][i]) & (location[vehicle_arange, next_hop_t] != -1)
            vehicles_action[i, action_vehicle_mask] = location[vehicle_arange[action_vehicle_mask], next_hop_t[action_vehicle_mask]]
        
        return {"vehicle": vehicles_action, "request": requests_action}, rnn_states_actor
    
    def eval(self):
        pass