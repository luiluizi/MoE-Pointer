from enum import Enum

import pandas as pd
import numpy as np
import torch
import tree
import geohash

from .multiagentenv import MultiAgentEnv

occupancy_logged = False

obs_func_map = {}
trans_func_map = {}

def unwrap_tensor(x):
    if isinstance(x, torch.Tensor):
        if x.dtype in [torch.int, torch.int32, torch.int64, torch.bool]:
            if x.numel() == 1:
                return x.item()
            else:
                return x.cpu().tolist()
        else:
            return x.cpu()
    else:
        return x

class VRR(Enum):
    """
    Vehicle Request Relation
    """
    no_relation = 0
    to_pickup = 1 # 这个值在 reassign 设定下没有用
    to_delivery = 2
    deliveryed = 3
    padded = 4 # 不能被 embedding
    skipped = 5


@torch.no_grad()
def floyd(g: torch.Tensor):
    for k in range(g.shape[-1]):
        g = torch.min(g, g[:, :, k].unsqueeze(-1) + g[:, k, :].unsqueeze(-2))
    return g

class DiscreteMVDPDP(MultiAgentEnv):
    """
    Discrete Multi Vehicle Dynamic Pickup Delivery Problem.
    允许 reassign, 提前分配无意义, 不使用 to_pickup
    requests choose vehicle to assign
    vehicle choose node as next-hop
    for simplification, vehicle mission can't be interruptable.


    Other Specializations:
    - 距离矩阵转为最段路矩阵
    - Discrete Multi VRP Problem
    - Discrete Multi Dial-a-Ride Problem.

    TODO:
    Add time window support.
    considering OD distribution.
    use Hyper Graph encode feature.
    use single node/ dual node represent a requests.
    Discrete and Contiguous Diff.
    Make use of historical data.
    Imitation Learning, when requests distribution is independent to state/action.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env_args = kwargs["env_args"]
        self.batch_size = kwargs["sample_batch_size"]
        self.use_sp = self.env_args["use_sp"]
        self.deliveryed_visible = self.env_args["deliveryed_visible"]
        self.info_batch_size = self.batch_size if self.env_args["info_batch_size"] == -1 else self.env_args["info_batch_size"]
        self.debug = self.env_args["debug"] # all envs use one seed.
        self.algorithm = kwargs["algorithm"]
        # Algorithm parameters
        self.use_tsp = kwargs["use_tsp"]
        self.use_ar = kwargs["use_ar"]

        self.dist_distribution = self.env_args["dist_distribution"]
        assert self.dist_distribution in ["random", "2D", "points"]
        self.interruptable = self.env_args["interruptable"]
        assert self.interruptable is False
        # max_value: 规定时间内获得最高的价值
        # min_dist: 完成所有任务的路程最少，可能有完不成的风险。可能造成智能体不动的情况，如何通过加 mask 解决。
        self.dist_cost = self.env_args["dist_cost"]

        self.n_vehicle = self.env_args["n_vehicle"]
        self.n_node = self.env_args["n_node"]
        self.n_tot_requests = self.env_args["n_init_requests"] + self.env_args["n_norm_requests"]
        self.max_consider_requests = self.n_tot_requests if self.env_args["max_consider_requests"] == -1 else self.env_args["max_consider_requests"]

        self.device = kwargs["device"]
        self.rng = torch.Generator(self.device)

        self.batch_arange = torch.arange(self.batch_size, device=self.device)
        self.node_arange = torch.arange(self.n_node, device=self.device)
        self.vehicle_arange = torch.arange(self.n_vehicle, device=self.device)
        self.requests_arange = torch.arange(self.n_tot_requests, device=self.device)

    def generate_non_overlapping(self, N, num_samples):
        prob_mat = torch.full([N, N], 1/(N*N-N), device=self.device)
        prob_mat.fill_diagonal_(0)
        index_flattened = torch.multinomial(prob_mat.flatten(), num_samples=num_samples, generator=self.rng, replacement=True)
        index_i = index_flattened // N
        index_j = index_flattened % N
        assert (index_i != index_j).all()
        return index_i, index_j
    
    def reset(self, **kwargs):
        if self.debug:
            self.rng.manual_seed(114514)
            print("use debug seed 114514.")
        
        # observable states
        self.vehicles = {
            "target": torch.randint(0, self.n_node, [self.batch_size, self.n_vehicle], generator=self.rng, device=self.device) if not self.debug else \
                torch.randint(0, self.n_node, [1, self.n_vehicle], generator=self.rng, device=self.device).repeat(self.batch_size, 1),
            "capacity": torch.full([self.batch_size, self.n_vehicle], self.env_args["max_capacity"], device=self.device),
            "space": torch.full([self.batch_size, self.n_vehicle], self.env_args["max_capacity"], device=self.device),
            "time_left": torch.full([self.batch_size, self.n_vehicle], 0, device=self.device),
            "cost": torch.zeros([self.batch_size, self.n_vehicle], dtype=torch.int64, device=self.device),
            "value": torch.zeros([self.batch_size, self.n_vehicle], dtype=torch.int64, device=self.device),
            "value_count": torch.zeros([self.batch_size, self.n_vehicle], dtype=torch.int64, device=self.device),
        }

        # unsymmetric matrix, digonal filled with 0.
        if self.dist_distribution == "random":
            node_node = torch.randint(1, self.env_args["max_dist"] + 1, [self.batch_size, self.n_node, self.n_node], generator=self.rng, device=self.device) if not self.debug else \
                torch.randint(1, self.env_args["max_dist"] + 1, [1, self.n_node, self.n_node], generator=self.rng, device=self.device).repeat(self.batch_size, 1, 1)
            if self.use_sp:
                node_node = floyd(node_node) # 平均路径长度折半
            if self.algorithm == "mapdp":
                rng2 = torch.Generator(self.device)
                rng2.manual_seed(0)
                node_coord = torch.randn(self.batch_size, self.n_node, 2, generator=rng2, device=self.device)
            else:
                node_coord = torch.zeros(self.batch_size, self.n_node, 2, device=self.device)
        elif self.dist_distribution == "points":
            _point_dim = 10
            points = torch.randn(self.batch_size, self.n_node, _point_dim, device=self.device) * self.env_args["max_dist"] / _point_dim
            node_node_f = (points[:, :, None] - points[:, None, :]).norm(p=2, dim=-1)
            node_node = node_node_f.ceil().type(torch.int64).clamp(min=1, max=self.env_args["max_dist"] + 1)
            # import ipdb;ipdb.set_trace()
            if self.algorithm == "mapdp":
                rng2 = torch.Generator(self.device)
                rng2.manual_seed(0)
                node_coord = torch.randn(self.batch_size, self.n_node, 2, generator=rng2, device=self.device)
            else:
                node_coord = torch.zeros(self.batch_size, self.n_node, 2, device=self.device)
        elif self.dist_distribution == "2D":
            import math
            _H = math.ceil(math.sqrt(self.n_node))
            node_coord = torch.stack((torch.arange(self.n_node, device=self.device) // _H, torch.arange(self.n_node, device=self.device) % _H), 1)
            node_node = (node_coord[:, None] - node_coord[None, :]).abs().sum(-1).repeat(self.batch_size, 1, 1)
            node_coord = node_coord.repeat(self.batch_size, 1, 1)

        self.nodes = {
            "coord": node_coord,
        }
    
        node_node[:, self.node_arange, self.node_arange] = 0
        if not self.debug:
            requests_from, requests_to = zip(*[
                self.generate_non_overlapping(self.n_node, self.n_tot_requests)
                for _ in range(self.batch_size)
            ])
        else:
            requests_from, requests_to = zip(*[
                self.generate_non_overlapping(self.n_node, self.n_tot_requests)
            ] * self.batch_size)
            
        requests_from = torch.stack(requests_from)
        requests_to = torch.stack(requests_to)
        visible = torch.full([self.batch_size, self.env_args["n_init_requests"] + self.env_args["n_norm_requests"]], False, device=self.device)
        visible[:, :self.env_args["n_init_requests"]] = True
        self.requests = {
            "from": requests_from,
            "to": requests_to,
            "value": node_node[self.batch_arange[:, None], requests_from, requests_to],
            "volumn": torch.ones(self.batch_size, self.n_tot_requests , dtype=torch.int64, device=self.device),
            "visible": visible
        }

        # theoretical server rate
        global occupancy_logged
        if not occupancy_logged:
            _all_workload = self.requests["value"].float().mean() * (self.env_args["n_init_requests"] + self.env_args["n_norm_requests"])
            _max_capacity = self.n_vehicle * self.env_args["n_frame"] # * self.env_args["max_capacity"]
            print("theoretical server rate:", (_max_capacity / _all_workload).item())
            occupancy_logged = True

        self._global = {
            "frame": 0,
            "n_exist_requests": torch.full([self.batch_size], self.env_args["n_init_requests"], device=self.device),
            "node_node": node_node,
            # 已完成的不可见，时间戳之后的不可见
            "vehicle_request": torch.full([self.batch_size, self.n_vehicle, self.n_tot_requests], VRR.no_relation.value, device=self.device)
        }

        # unobservable states
        assert self.env_args["n_requested_frame"] <= self.env_args["n_frame"]
        self.frame_n_accumulate_requests = torch.randint(
            self.env_args["n_init_requests"], self.env_args["n_init_requests"] + self.env_args["n_norm_requests"] + 1, [self.batch_size, self.env_args["n_requested_frame"] - 1], generator=self.rng, device=self.device
        ).sort().values if not self.debug else \
            torch.randint(
                self.env_args["n_init_requests"], self.env_args["n_init_requests"] + self.env_args["n_norm_requests"] + 1, [1, self.env_args["n_requested_frame"] - 1], generator=self.rng, device=self.device
            ).repeat(self.batch_size, 1).sort().values
        self.frame_n_accumulate_requests = torch.cat((
            torch.tensor([self.env_args["n_init_requests"]], device=self.device).expand(self.batch_size, 1), 
            self.frame_n_accumulate_requests, 
            torch.full([self.batch_size, self.env_args["n_frame"] - self.env_args["n_requested_frame"] + 1], self.n_tot_requests, device=self.device)
        ), dim=1)
        assert self.frame_n_accumulate_requests.shape[1] == self.env_args["n_frame"] + 1

        self.requests["appear"] = (self.requests_arange[None, :, None] < self.frame_n_accumulate_requests[:, None, :]).to(torch.int64).argmax(-1)

        self.pre_obs, self.pre_unobs = self.get_obs()
        return self.pre_obs
    
    @torch.no_grad()
    def get_obs(self):
        # be careful for inplace operation, which need's clone for save pre_obs.
        # cur_max_consider_requests = min(self.env_args["max_consider_requests"], self._global["n_exist_requests"].max())
        cur_max_consider_requests = self.max_consider_requests
        visible = self.requests["visible"]
        n_exist_requests = visible.sum(-1)
        n_consider_requests = n_exist_requests.clamp_max(cur_max_consider_requests)

        if self.algorithm in ["mapt"]:
            requests_meet_vehicle = ((self.vehicles["target"][:, None, :] == self.requests["from"][:, :, None]) & (self.vehicles["time_left"] == 0)[:, None, :]).any(-1) & (self._global["vehicle_request"] == 0).all(-2)
            requests_on_vehicle = (((self._global["vehicle_request"] == 1) | (self._global["vehicle_request"] == 2)) & (self.vehicles["time_left"] == 0)[:, :, None]).any(-2)
            _index = ((visible * 4 + requests_on_vehicle * 2 + requests_meet_vehicle * 1) * self.n_tot_requests + self.n_tot_requests - self.requests_arange).argsort(-1, descending=True)
            extractor = _index[:, :cur_max_consider_requests]
        else:
            # TODO: 把 no_relation 放在 to_pickup/ to_delivery 前面。但是这样会影响位置编码，暂时先不改了。
            extractor = torch.zeros(self.batch_size, cur_max_consider_requests, dtype=torch.int64, device=self.device)
            for i in range(self.batch_size):
                _index = torch.where(visible[i])[0][:cur_max_consider_requests]
                extractor[i, :_index.shape[0]] = _index
            
        vehicle_request = self._global["vehicle_request"].gather(2, extractor[:, None, :].expand(-1, self.n_vehicle, -1))
        unpadded_requests_mask = torch.arange(cur_max_consider_requests, device=self.device) < n_consider_requests[:, None]
        vehicle_request.masked_fill_(~unpadded_requests_mask[:, None, :], VRR.padded.value)

        obs = {
            "nodes": {
                "coord": self.nodes["coord"],
                "n_vehicle": torch.stack([torch.bincount(t1, minlength=self.n_node) for t1 in self.vehicles["target"]]),
                "n_requests_from": torch.stack([torch.bincount(t1[t2], minlength=self.n_node) for t1, t2 in zip(self.requests["from"], visible)]),
                "n_requests_to": torch.stack([torch.bincount(t1[t2], minlength=self.n_node) for t1, t2 in zip(self.requests["to"], visible)]),
            },
            "vehicles": {
                "target": self.vehicles["target"].clone(), # inplace update
                "capacity": self.vehicles["capacity"],
                "space": self.vehicles["space"].clone(), # inplace update
                "time_left": self.vehicles["time_left"].clone(), # inplace update
                # agent don't need to observation cost and value.
            },
            "requests": {
                "from": self.requests["from"].gather(-1, extractor),
                "to": self.requests["to"].gather(-1, extractor),
                "value": self.requests["value"].gather(-1, extractor),
                "volumn": self.requests["volumn"].gather(-1, extractor),
                "appear": self.requests["appear"].gather(-1, extractor), # should not be used by only algorithms.
            },
            "global": {
                "frame": torch.full([self.batch_size], self._global["frame"], device=self.device), # int
                "n_exist_requests": n_exist_requests,
                "n_consider_requests": n_consider_requests,
                "node_node": self._global["node_node"],
                "vehicle_request": vehicle_request.clone(), # inplace update
                # "n_unoccur_requests": self.n_tot_requests - self.frame_n_accumulate_requests[self.batch_arange, self._global["frame"]],
            },
            "extractor": extractor,
        }
        unobs = {
            "extractor": extractor,
            "unpadded_requests_mask": unpadded_requests_mask,
        }

        if self.algorithm in obs_func_map:
            obs_func_map[self.algorithm](self, obs, unobs)

        return obs, unobs
    
    @torch.no_grad()
    def step(self, actions):
        # padded requests 和暂时不分配的 requests 被解码为 self.n_vehicle。
        # TODO: Add bound detector. Currently, requests can't be all deliveried.
        # TODO: 检测当前没有可以被 delivery 的物品
        infos, debug_infos = {}, {}

        extractor = self.pre_unobs["extractor"]

        preassign = actions.get("preassign", None)
        if preassign is not None:
            assert preassign.shape == extractor.shape
            preassign_mask = preassign != self.n_vehicle
            assert (preassign_mask <= (self.pre_obs["global"]["vehicle_request"]==0).all(dim=-2)).all(), "only `no_relation` requests could be pre-assigned."
            preassign_batch = self.batch_arange[:, None].masked_select(preassign_mask)
            preassign_request = extractor[preassign_mask]
            preassign_vehicle = preassign[preassign_mask]
            assert (self._global["vehicle_request"][preassign_batch, :, preassign_request] == 0).all(), "double check, only `no_relation` requests could be pre-assigned."
            self._global["vehicle_request"][preassign_batch, preassign_vehicle, preassign_request] = VRR.to_pickup.value

        considering_may_assign = actions["request"] # B x min(n_exist_requests.max(), max_consider_requests))
        assert considering_may_assign.shape == extractor.shape
        considering_assigning_mask = considering_may_assign != self.n_vehicle
        assert (considering_assigning_mask <= (self.pre_obs["global"]["vehicle_request"]<=1).all(dim=-2)).all(), "only `no_relation` requests shold be assigned."
        pickuping_batch = self.batch_arange[:, None].masked_select(considering_assigning_mask)
        pickuping_request = extractor[considering_assigning_mask]
        pickuping_vehicle = considering_may_assign[considering_assigning_mask]
        if self.info_batch_size != 0:
            debug_infos["frame"] = self._global["frame"]
            debug_infos["space"] = self.vehicles["space"][:self.info_batch_size]
            debug_infos["can_pickup"] = ((self.vehicles["target"][:, :, None] == self.requests["from"][:, None, :]) & (self._global["vehicle_request"] == 0).all(dim=1, keepdim=True) & (self.vehicles["time_left"] == 0)[:, :, None] & (self.vehicles["space"][:, :, None] >= self.requests["volumn"][:, None, :]) & self.requests["visible"][:, None]).any(-2)[:self.info_batch_size].sum() // self.info_batch_size
            debug_infos["pickuping"] = extractor[:self.info_batch_size][considering_assigning_mask[:self.info_batch_size]].numel() // self.info_batch_size
            debug_infos["onway"] = (self.vehicles["time_left"] != 0)[:self.info_batch_size].sum() // self.info_batch_size
            debug_infos["visible"] = self.requests["visible"][:self.info_batch_size].sum() // self.info_batch_size
        # 检查 pickup 的车辆和 request 在同一点
        assert (self.vehicles["time_left"][pickuping_batch, pickuping_vehicle] == 0).all()
        assert (self.vehicles["target"][pickuping_batch, pickuping_vehicle] == self.requests["from"][pickuping_batch, pickuping_request]).all()
        assert (self._global["vehicle_request"][pickuping_batch, :, pickuping_request]<=1).all(), "double check, only `no_relation` requests shold be assigned."
        # 如果存在 preassign，不能抢占
        preassigned = (self._global["vehicle_request"] == 1).any(-2)
        preassigned_request_mask = preassigned[pickuping_batch, pickuping_request]
        assert (self._global["vehicle_request"][pickuping_batch[preassigned_request_mask], :, pickuping_request[preassigned_request_mask]].sum(-1) == 1).all()
        preassigned_vehicle = torch.where(self._global["vehicle_request"][pickuping_batch[preassigned_request_mask], :, pickuping_request[preassigned_request_mask]])[1]
        assert (pickuping_vehicle[preassigned_request_mask] == preassigned_vehicle).all()
        # End check preassign
        self._global["vehicle_request"][pickuping_batch, pickuping_vehicle, pickuping_request] = VRR.to_delivery.value
        assert ((self._global["vehicle_request"] == VRR.to_delivery.value).sum(-1) <= self.vehicles["capacity"]).all(), "vehicle capacity exceeded."

        # TODO: 不需要解码的 vehicle。其 target 设置为之前的 target.
        vehicle_to = actions["vehicle"] # B x n_vehicle
        assert (vehicle_to < self.n_node).all()
        assert ((self.vehicles["time_left"] != 0) <= (vehicle_to == self.vehicles["target"])).all()
        dispatching_mask = self.vehicles["time_left"] == 0
        self.vehicles["time_left"][dispatching_mask] = self._global["node_node"][self.batch_arange[:, None].masked_select(dispatching_mask), self.vehicles["target"][dispatching_mask], vehicle_to[dispatching_mask]]
        assert (vehicle_to[dispatching_mask] < self.n_node).all()
        self.vehicles["target"][dispatching_mask] = vehicle_to[dispatching_mask] # 顺序不能错

        current_cost = (self.vehicles["time_left"] != 0).sum(-1)
        self.vehicles["cost"] += self.vehicles["time_left"] != 0
        delivering_mask = self.vehicles["time_left"] == 1 # time_left==0 和 time_left==1 互不影响
        delivering = delivering_mask[:, :, None] & (self._global["vehicle_request"] == VRR.to_delivery.value) & (self.vehicles["target"][:, :, None] == self.requests["to"][:, None, :])
        delivering_value = delivering * self.requests["value"][:, None, :]
        current_value = delivering_value.sum(dim=-1).sum(dim=-1)
        self.vehicles["value"] += delivering_value.sum(dim=-1)
        self.vehicles["value_count"] += delivering.sum(dim=-1)
        self._global["vehicle_request"][delivering] = VRR.deliveryed.value # 可能一个 step 就完成 pickup delivery 了
        if self.info_batch_size != 0:
            debug_infos["delivering"] = delivering[:self.info_batch_size].sum() // self.info_batch_size
        
        self.vehicles["time_left"] = (self.vehicles["time_left"] - 1).clamp_min(0)
        self.vehicles["space"] = (self.vehicles["capacity"] - ((self._global["vehicle_request"]==VRR.to_delivery.value)*self.requests["volumn"][:, None, :]).sum(-1))
        assert (self.vehicles["space"] >= 0).all()
        self._global["frame"] += 1

        # removal complete requests and generate new requests.
        new_request_l, new_request_r = self.frame_n_accumulate_requests[self.batch_arange, self._global["frame"] - 1], self.frame_n_accumulate_requests[self.batch_arange, self._global["frame"]]
        self.requests["visible"][self.requests_arange < new_request_l[:, None]] ^= True
        self.requests["visible"][self.requests_arange < new_request_r[:, None]] ^= True
        if not self.deliveryed_visible:
            self.requests["visible"][(self._global["vehicle_request"]==VRR.deliveryed.value).any(dim=-2)] = False
        
        if self.algorithm in trans_func_map:
            trans_func_map[self.algorithm](self, actions)

        obs, unobs = self.get_obs()
        rewards = current_value - current_cost * self.dist_cost
        dones = torch.full([self.batch_size], self._global["frame"], device=self.device) == self.env_args["n_frame"]
        if self.algorithm in ["mapt", "prob_heuristic"] and self.use_tsp and self.use_ar:
            assert (~dones[:, None] | (self.vehicles["space"] == self.vehicles["capacity"])).all()

        self.pre_obs = obs
        self.pre_unobs = unobs

        if self.info_batch_size != 0:
            debug_infos["current_finish_ratio"] = (self.vehicles["value_count"].sum(-1) / self.frame_n_accumulate_requests[self.batch_arange, self._global["frame"]])[:self.info_batch_size].mean()
            debug_infos["global_finish_ratio"] = (self.vehicles["value_count"].sum(-1) / self.frame_n_accumulate_requests[:, -1])[:self.info_batch_size].mean()
        infos["current_finish_ratio"] = self.vehicles["value_count"].sum(-1) / self.frame_n_accumulate_requests[self.batch_arange, self._global["frame"]]
        infos["global_finish_ratio"] = self.vehicles["value_count"].sum(-1) / self.frame_n_accumulate_requests[:, -1]

        if self.info_batch_size != 0:
            print(tree.map_structure(unwrap_tensor, debug_infos))

        # final check, 保证每个 request 只对最多一辆车有 to_pickup, pickuped, deliveried 状态。 
        assert (sum(self._global["vehicle_request"] == t for t in range(1, VRR.deliveryed.value + 1)).sum(-2) <= 1).all()
        return obs, rewards, dones, infos

    def get_env_info(self):
        env_info = {
            "observation_spaces": self.observation_space,
            "action_spaces": self.action_space
        }
        return env_info
    
    def seed(self, seed):
        self.rng.manual_seed(seed)

class DiscreteMVDPDPDHRD(DiscreteMVDPDP):
    
    def __init__(self, **kwargs):
        self.env_args = kwargs["env_args"]
        self.deliveryed_visible = self.env_args["deliveryed_visible"]
        self.info_batch_size = self.batch_size if self.env_args["info_batch_size"] == -1 else self.env_args["info_batch_size"]
        self.debug = self.env_args["debug"] # all envs use one seed.
        self.algorithm = kwargs["algorithm"]
        self.use_tsp = kwargs["use_tsp"]
        self.use_ar = kwargs["use_ar"]

        self.place = kwargs["place"]
        self.is_train = kwargs["is_train"]
        self.suffix = kwargs["suffix"]
        self.pack_size = self.env_args["pack_size"]

        self.interruptable = self.env_args["interruptable"]
        assert self.interruptable is False
        # max_value: 规定时间内获得最高的价值
        # min_dist: 完成所有任务的路程最少，可能有完不成的风险。可能造成智能体不动的情况，如何通过加 mask 解决。
        self.dist_cost = self.env_args["dist_cost"]

        self.config = {
            "tw": {
                "n_node": 36,
                "n_vehicle": 20,
                "n_tot_requests": 1000,
                "except": [],
            },
            "sg": {
                "n_node": 36,
                "n_vehicle": 15,
                "n_tot_requests": 700,
                "except": []
            },
            "se": {
                "n_node": 36,
                "n_vehicle": 7,
                "n_tot_requests": 150,
                "max_consider_requests": 150,
                "except": ['u7xqm']
            },
            
        }[self.place]
        self.n_node = self.config["n_node"]
        self.n_tot_requests = self.config["n_tot_requests"]
        self.n_vehicle = self.config["n_vehicle"]

        # Dataset specific parameter
        self.env_args["n_node"] = self.n_node
        self.env_args["n_vehicle"] = self.n_vehicle
        self.env_args["n_tot_requests"] = self.n_tot_requests
        if "max_consider_requests" in self.config:
            self.env_args["max_consider_requests"] = self.config["max_consider_requests"]
        self.max_consider_requests = self.n_tot_requests if self.env_args["max_consider_requests"] == -1 else self.env_args["max_consider_requests"]

        
        self.device = kwargs["device"]
        self.rng = torch.Generator(self.device)

        self.genearte()

        if self.is_train:
            self.batch_size = kwargs["sample_batch_size"]
        else:
            self.batch_size = min(self.cached["requests_from"].shape[0], kwargs["sample_batch_size"])
        self.batch_arange = torch.arange(self.batch_size, device=self.device)
        self.node_arange = torch.arange(self.n_node, device=self.device)
        self.vehicle_arange = torch.arange(self.n_vehicle, device=self.device)
        self.requests_arange = torch.arange(self.n_tot_requests, device=self.device)

    def genearte(self):
        delta = 0.02197265625
        interval = 30 # minutes
        n_node = self.n_node
        except_node = self.config["except"]

        assert n_node == self.env_args["n_node"]

        orders = pd.read_csv(f"data/data_{self.place}/orders_{self.place}_{self.suffix}.txt", index_col=0)
        vendors = pd.read_csv(f"data/data_{self.place}/vendors_{self.place}.txt", index_col=0)

        vendors_geohash = vendors.geohash.value_counts().index
        nodes = orders.geohash.value_counts().drop(except_node).drop(vendors_geohash)[:n_node - len(vendors_geohash)].index.tolist() + vendors_geohash.tolist()
        assert n_node == len(nodes)
        nodes.sort()
        
        assert len(nodes) == n_node
        nodes_coords = np.array([geohash.decode(code) for code in nodes])
        for code in nodes:
            _, _, dlat, dlon = geohash.decode(code, True)
            assert dlat == delta and dlon == delta
        dist = (np.abs(nodes_coords[:, None] - nodes_coords[None]).sum(-1) / delta + 0.5).astype(int)
        assert (dist % 2 == 0).all()
        dist = dist // 2

        orders = orders.join(vendors[["vendor_id", "geohash"]].set_index("vendor_id"), on="vendor_id", how="left", rsuffix="_v")
        assert vendors.geohash.isin(nodes).all()

        orders = orders.drop(orders.index[~orders.geohash.isin(nodes)])
        orders = orders.drop(orders.index[orders.geohash==orders.geohash_v])
        orders = orders.reset_index()

        orders["order_time"] = pd.to_datetime(orders["order_time"], format="%H:%M:%S")
        orders["from"] = orders["geohash_v"].map({code: idx for idx, code in enumerate(nodes)})
        orders["to"] = orders["geohash"].map({code: idx for idx, code in enumerate(nodes)})

        assert 24 * 60 % interval == 0
        n_norequest_frame = 1
        n_requested_frame = 24 * 60 // interval
        n_frame = n_requested_frame + n_norequest_frame
        self.n_frame = n_frame
        assert self.n_frame == self.env_args["n_frame"]

        groups = list(orders.groupby(orders["order_day"]))
        n_inst = len(groups)

        node_node = (np.abs(nodes_coords[:, None] - nodes_coords[None]).sum(-1) / delta + 0.5).astype(int)
        assert (node_node % 2 == 0).all()
        node_node = node_node // 2
        self.node_node = node_node
        assert node_node.max() <= self.env_args["max_dist"]

        requests_from_batched = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        requests_to_batched = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        requests_value_batched = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        frame_n_accumulate_requests_batched = []

        for inst_idx, (_, day_df) in enumerate(orders.groupby(orders["order_day"])):
            requests_from = []
            requests_to = []
            requests_value = []
            requests_occur = []

            for (_from, _to), uv_df in day_df.groupby(["from", "to"]):
                uv_df = uv_df.sort_values("order_time")
                minutes = uv_df["order_time"].dt.hour * 60 + uv_df["order_time"].dt.minute
                last = -1
                for i in range(len(uv_df)):
                    if i - last >= self.pack_size:
                        requests_from.append(_from)
                        requests_to.append(_to)
                        requests_value.append(i - last)
                        requests_occur.append(minutes.iloc[i] // interval)
                        last = i
                # we don't consider tail requests.

            requests_occur = torch.tensor(requests_occur)[:self.n_tot_requests]            
            reindex = requests_occur.argsort()
            requests_from = torch.tensor(requests_from)[:self.n_tot_requests][reindex]
            requests_to = torch.tensor(requests_to)[:self.n_tot_requests][reindex]
            requests_value = torch.tensor(requests_value)[:self.n_tot_requests][reindex]
            requests_occur = requests_occur[reindex]
            frame_n_accumulate_requests_batched.append(torch.bincount(requests_occur, minlength=self.n_frame + 1))
            
            requests_from_batched[inst_idx, :requests_from.shape[0]] = requests_from.to(self.device)
            requests_to_batched[inst_idx, :requests_from.shape[0]] = requests_to.to(self.device)
            requests_value_batched[inst_idx, :requests_from.shape[0]] = requests_value.to(self.device)

        frame_n_accumulate_requests_batched = torch.stack(frame_n_accumulate_requests_batched).to(self.device)
        frame_n_accumulate_requests_batched = frame_n_accumulate_requests_batched.cumsum(-1)

        print("n_requests", frame_n_accumulate_requests_batched[:, -1].cpu())
        print("generating instances finish.")

        self.cached = {}
        self.cached["coord"] = torch.from_numpy(nodes_coords).to(self.device)
        self.cached["requests_from"] = requests_from_batched
        self.cached["requests_to"] = requests_to_batched
        self.cached["requests_value"] = requests_value_batched
        self.cached["requests_volumn"] = torch.ones_like(requests_value_batched)

        self.cached["node_node"] = torch.from_numpy(node_node).to(self.device)
        self.cached["frame_n_accumulate_requests"] = frame_n_accumulate_requests_batched

    def reset(self, **kwargs):
        if self.is_train:
            batch_idx = torch.randint(0, self.cached["requests_from"].shape[0], (self.batch_size, ), generator=self.rng, device=self.device)
        else:
            rng2 = torch.Generator(self.device)
            rng2.manual_seed(0)
            batch_idx = torch.randperm(self.cached["requests_from"].shape[0], generator=rng2, device=self.device)[:self.batch_size]
            # assert self.cached["requests_from"].shape[0] == self.batch_size
        
        self.nodes = {
            "coord": self.cached["coord"].repeat(self.batch_size, 1, 1),
        }

        # observable states
        self.vehicles = {
            "target": torch.randint(0, self.n_node, [self.batch_size, self.n_vehicle], generator=self.rng, device=self.device),
            "capacity": torch.full([self.batch_size, self.n_vehicle], self.env_args["max_capacity"], device=self.device),
            "space": torch.full([self.batch_size, self.n_vehicle], self.env_args["max_capacity"], device=self.device),
            "time_left": torch.full([self.batch_size, self.n_vehicle], 0, device=self.device),
            "cost": torch.zeros([self.batch_size, self.n_vehicle], dtype=torch.int64, device=self.device),
            "value": torch.zeros([self.batch_size, self.n_vehicle], dtype=torch.int64, device=self.device),
            "value_count": torch.zeros([self.batch_size, self.n_vehicle], dtype=torch.int64, device=self.device),
        }

        visible = torch.full([self.batch_size, self.n_tot_requests], False, device=self.device)
        visible[self.requests_arange < self.cached["frame_n_accumulate_requests"][batch_idx, :1]] = True
        self.requests = {
            "from": self.cached["requests_from"][batch_idx],
            "to": self.cached["requests_to"][batch_idx],
            "value": self.cached["requests_value"][batch_idx],
            "volumn": self.cached["requests_volumn"][batch_idx],
            "visible": visible,
            "appear": (self.requests_arange[None, :, None] < self.cached["frame_n_accumulate_requests"][batch_idx, None, :]).to(torch.int64).argmax(-1)
        }

        rel_mat = torch.full([self.batch_size, self.n_vehicle, self.n_tot_requests], VRR.no_relation.value, device=self.device)
        rel_mat.masked_fill_((self.requests_arange >= self.cached["frame_n_accumulate_requests"][batch_idx, -1:])[:, None, :], VRR.padded.value)
        self._global = {
            "frame": 0,
            "n_exist_requests": self.cached["frame_n_accumulate_requests"][batch_idx, 0],
            "node_node": self.cached["node_node"].repeat(self.batch_size, 1, 1), 
            "vehicle_request" : rel_mat,
        }

        self.frame_n_accumulate_requests = self.cached["frame_n_accumulate_requests"][batch_idx]

        # import ipdb;ipdb.set_trace()
        # theoretical server rate
        global occupancy_logged
        if not occupancy_logged:
            _all_workload = self._global["node_node"][self.batch_arange[:, None], self.requests["from"], self.requests["to"]].float().sum(-1).mean()
            _max_capacity = self.n_vehicle * self.env_args["n_frame"]# * self.env_args["max_capacity"]
            print("theoretical server rate:", (_max_capacity / _all_workload).item())
            occupancy_logged = True

        self.pre_obs, self.pre_unobs = self.get_obs()
        return self.pre_obs

