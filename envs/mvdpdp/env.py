from enum import Enum
import torch

from .multiagentenv import MultiAgentEnv
from utils.util import get_logger
import math

occupancy_logged = False

obs_func_map = {}
trans_func_map = {}

# 随机数生成器放置在cpu上以保证跨显卡架构结果的一致性

class RELATION(Enum):
    no_relation = 0
    to_pickup = 1
    to_delivery = 2
    deliveryed_stage1 = 3
    deliveryed_stage2 = 4
    deliveryed_stage3 = 5
    padded = 6 # 不能被 embedding
    skipped = 7 # 可以解码但是由于策略设置被跳过

@torch.no_grad()
def floyd(g: torch.Tensor):
    for k in range(g.shape[-1]):
        g = torch.min(g, g[:, :, k].unsqueeze(-1) + g[:, k, :].unsqueeze(-2))
    return g

class DroneTransferEnv(MultiAgentEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.env_args = kwargs["env_args"]
        self.batch_size = kwargs["batch_size"]
        self.use_sp = self.env_args["use_sp"]
        self.deliveryed_visible = self.env_args["deliveryed_visible"]
        self.info_batch_size = self.batch_size if self.env_args["info_batch_size"] == -1 else self.env_args["info_batch_size"]
        self.debug = self.env_args["debug"] # all envs use one seed.
        self.algorithm = kwargs["algorithm"]

        self.dist_distribution = self.env_args["dist_distribution"]
        print("dist_distribution", self.dist_distribution)
        assert self.dist_distribution == "2D", "dist_distribution must be '2D'"
       
        self.dist_cost = self.env_args["dist_cost"]

        # 环境实体设置
        self.n_node = self.env_args["n_node"]
        self.n_station = self.env_args["n_station"]
        self.n_courier = self.env_args["n_courier"]
        self.n_drone = self.env_args["n_drone"] 
        self.min_direct_dist = self.env_args['min_direct_dist'] # 设置订单的最短距离
        self.drone_speed_ratio = self.env_args.get('drone_speed_ratio', 4)
        
        # 载具cost以及订单profit设置
        self.drone_cost_ratio = self.env_args['dist_cost_drone']
        self.courier_cost_ratio = self.env_args['dist_cost_courier']
        self.request_profit_ratio = self.env_args['dist_req_profit']
        
        # self.algorithm = "BR"
        self.n_tot_requests = self.env_args["n_init_requests"] + self.env_args["n_norm_requests"]
        self.max_consider_requests = self.n_tot_requests if self.env_args["max_consider_requests"] == -1 else self.env_args["max_consider_requests"]
        self.device = kwargs["device"]
        self.rng = torch.Generator('cpu')
        self.rng_env = torch.Generator('cpu')

        self.batch_arange = torch.arange(self.batch_size, device=self.device)
        self.node_arange = torch.arange(self.n_node, device=self.device)
        self.requests_arange = torch.arange(self.n_tot_requests, device=self.device)
        # self.logger = get_logger()

    def generate_non_overlapping(self, num_samples, node_node, min_direct_dist=0):
        non_station_idx = torch.nonzero(~self.station_mask[0], as_tuple=False).reshape(-1)
        n_non_station = non_station_idx.numel()
        assert n_non_station >= 2, "非中转站节点数不足，无法生成订单"
        
        non_station_dist = node_node[non_station_idx][:, non_station_idx]
        valid_mask = (non_station_dist >= min_direct_dist) & ~torch.eye(
            n_non_station, dtype=torch.bool, device=self.device
        )
        valid_pairs = torch.nonzero(valid_mask)
        selected_indices = torch.randint(
            low=0,
            high=len(valid_pairs),
            size=(num_samples,),
            generator=self.rng,
            device='cpu'
        ).to(self.device)
        selected_pairs = valid_pairs[selected_indices]
        
        all_i = non_station_idx[selected_pairs[:, 0]]
        all_j = non_station_idx[selected_pairs[:, 1]]
        return all_i, all_j
   

    def reset(self, **kwargs):
        if self.debug:
            self.rng.manual_seed(123321)
            print("use debug seed 123321.")

        if self.dist_distribution == "2D":
            n_node_tot = self.env_args["n_node_tot"]
            _H = math.ceil(math.sqrt(n_node_tot))
            grid_coord = torch.stack((
                torch.arange(n_node_tot, device=self.device) // _H,
                torch.arange(n_node_tot, device=self.device) % _H
            ), 1)
            n_node = self.n_node
            n_station = self.n_station
            # 随机选择 n_node 个节点
            node_indices = torch.randperm(n_node_tot, generator=self.rng_env, device='cpu')[:n_node].to(self.device)
            node_coord = grid_coord[node_indices]

            # 计算节点间的曼哈顿距离
            node_node = (node_coord[:, None] - node_coord[None, :]).abs().sum(-1)
            node_node = node_node.repeat(self.batch_size, 1, 1)
            node_coord = node_coord.repeat(self.batch_size, 1, 1)
            
            print("node average dist", torch.mean(node_node.float()).item())

            # 随机选择 n_station 个中转站
            station_indices = torch.randperm(n_node, generator=self.rng_env, device='cpu')[:n_station].to(self.device)

            station_idx = torch.cat([station_indices, torch.tensor([n_node], device=self.device)]).unsqueeze(0).repeat(self.batch_size, 1)
            station_mask = torch.zeros((self.batch_size, n_node), dtype=torch.bool, device=self.device)
            station_mask[:, station_indices] = True 

            # 计算中转站之间的距离
            submat = node_node[0, station_indices][:, station_indices]
            station_node_node = submat.unsqueeze(0).repeat(self.batch_size, 1, 1)
            
            import pandas as pd
            import matplotlib.pyplot as plt
            def save_nodes_and_stations(node_coord, station_idx, filepath="nodes_stations.csv"):
                batch_size, n_node, _ = node_coord.shape
                records = []
                for b in range(batch_size):
                    for i in range(n_node):
                        x, y = node_coord[b, i].tolist()
                        is_station = int(i in station_idx[b].tolist())  # 判断是否是站点
                        records.append({
                            "batch": b,
                            "node_id": i,
                            "x": x,
                            "y": y,
                            "is_station": is_station
                        })
                df = pd.DataFrame(records)
                df.to_csv(filepath, index=False)
                print(f"Saved nodes & stations info to {filepath}")
            
            def plot_nodes_and_stations(filepath="nodes_stations.csv", out_file="nodes_plot.png", batch_id=0):
                df = pd.read_csv(filepath)

                # 选取某一个 batch
                batch_df = df[df["batch"] == batch_id]

                # 普通节点
                nodes = batch_df[batch_df["is_station"] == 0]
                # 中转站
                stations = batch_df[batch_df["is_station"] == 1]

                plt.figure(figsize=(6, 6))
                plt.scatter(nodes["x"], nodes["y"], c="blue", label="Node", alpha=0.7)
                plt.scatter(stations["x"], stations["y"], c="red", label="Station", s=80, marker="s")

                for _, row in batch_df.iterrows():
                    plt.text(row["x"]+0.1, row["y"]+0.1, str(row["node_id"]), fontsize=8)

                plt.xlabel("X")
                plt.ylabel("Y")
                plt.title(f"Batch {batch_id}: Nodes and Stations")
                plt.legend()
                plt.grid(True)
                plt.axis("equal")
                plt.show()
                plt.savefig(out_file, dpi=300, bbox_inches="tight")
                plt.close()
            
            # save_nodes_and_stations(node_coord, station_idx)
            # plot_nodes_and_stations()
            # assert(False)

        self.nodes = { "coord": node_coord }
        self.node_node = node_node
        self.station_node_node = station_node_node
        self.station_mask = station_mask
        self.station_idx = station_idx

        node_node[:, self.node_arange, self.node_arange] = 0
        generate_func = self.generate_non_overlapping
        if not self.debug:
            requests_from, requests_to = zip(*[generate_func(self.n_tot_requests, self.node_node[i], self.min_direct_dist) for i in range(self.batch_size)])
        else:
            requests_from, requests_to = zip(*[generate_func(self.n_tot_requests, self.node_node[0], self.min_direct_dist)] * self.batch_size)
            
        requests_from = torch.stack(requests_from)
        requests_to = torch.stack(requests_to)    
        visible = torch.full([self.batch_size, self.env_args["n_init_requests"] + self.env_args["n_norm_requests"]], False, device=self.device)
        visible[:, :self.env_args["n_init_requests"]] = True

        # TODO courier初始分布
        
        # drone初始分布生成，尽量不集中在同一点
        if not self.debug:
            cycle_indices = torch.arange(self.n_drone, device=self.device) % self.n_station
            batch_offsets = torch.randint(0, self.n_station, (self.batch_size,), generator=self.rng, device='cpu').to(self.device)

            rand_idx = (cycle_indices.unsqueeze(0) + batch_offsets.unsqueeze(1)) % self.n_station
            drones_target = torch.gather(self.station_idx[:,:-1].unsqueeze(1).expand(-1, self.n_drone, -1), 2, rand_idx.unsqueeze(2)).squeeze(2)
        else:
            # debug 模式下，先生成 [1, n_drone] 的 rand_idx，再 repeat 到整个 batch
            rand_idx = torch.randint(0, self.n_station, (1, self.n_drone), generator=self.rng, device='cpu' ).to(self.device)
            drones_target = torch.gather(self.station_idx[:1].unsqueeze(1).expand(-1, self.n_drone, -1), 2, rand_idx.unsqueeze(2)).squeeze(2).repeat(self.batch_size, 1)

        # 设置初始状态
        self.couriers = {
            "target": torch.randint(0, self.n_node, [self.batch_size, self.n_courier], generator=self.rng, device='cpu').to(self.device) if not self.debug else \
                torch.randint(0, self.n_node, [1, self.n_courier], generator=self.rng, device='cpu').to(self.device).repeat(self.batch_size, 1),
            "capacity": torch.full([self.batch_size, self.n_courier], self.env_args["max_capacity"], device=self.device),
            "space": torch.full([self.batch_size, self.n_courier], self.env_args["max_capacity"], device=self.device),
            "time_left": torch.full([self.batch_size, self.n_courier], 0, device=self.device),
            "cost": torch.zeros([self.batch_size, self.n_courier], dtype=torch.int64, device=self.device),
            "value": torch.zeros([self.batch_size, self.n_courier], dtype=torch.int64, device=self.device),
            "value_count": torch.zeros([self.batch_size, self.n_courier], dtype=torch.int64, device=self.device),
        }
        self.couriers_stage1_value_count = torch.zeros([self.batch_size, self.n_courier], dtype=torch.int64, device=self.device)

        self.drones = {
            "target": drones_target,
            "space": torch.full([self.batch_size, self.n_drone], 1, device=self.device),
            "time_left": torch.full([self.batch_size, self.n_drone], 0, device=self.device),
            "float_time_left": torch.full([self.batch_size, self.n_drone], 0.0, device=self.device),
            "cost": torch.zeros([self.batch_size, self.n_drone], dtype=torch.int64, device=self.device),
            "value": torch.zeros([self.batch_size, self.n_drone], dtype=torch.int64, device=self.device),
            "value_count": torch.zeros([self.batch_size, self.n_drone], dtype=torch.int64, device=self.device),
        }
        self.requests = {
            "from": requests_from,
            "to": requests_to,
            "value": node_node[self.batch_arange[:, None], requests_from, requests_to] * self.request_profit_ratio,
            "volumn": torch.ones(self.batch_size, self.n_tot_requests , dtype=torch.int64, device=self.device),
            "visible": visible,
            "station1": torch.full([self.batch_size, self.n_tot_requests], self.n_node, device=self.device),
            "station2": torch.full([self.batch_size, self.n_tot_requests], self.n_node, device=self.device),
        }
        self._global = {
            "frame": 0,
            "n_exist_requests": torch.full([self.batch_size], self.env_args["n_init_requests"], device=self.device),
            "node_node": node_node,
            "station_idx": station_idx,
            "station_mask": station_mask,
            "station_node_node": station_node_node,
            "courier_request": torch.full([self.batch_size, self.n_courier, self.n_tot_requests], RELATION.no_relation.value, device=self.device),
            "drone_request": torch.full([self.batch_size, self.n_drone, self.n_tot_requests], RELATION.no_relation.value, device=self.device),
            "request_done": torch.full([self.batch_size, self.n_tot_requests], False, device=self.device),
        }
        
        # theoretical server rate 后面要去掉
        global occupancy_logged
        if not occupancy_logged:
            _all_workload = self.requests["value"].float().mean() * (self.env_args["n_init_requests"] + self.env_args["n_norm_requests"])
            _max_capacity = self.n_courier * self.env_args["n_frame"] # * self.env_args["max_capacity"]
            # self.logger.info(f"theoretical server rate: {(_max_capacity / _all_workload).item()}")
            occupancy_logged = True

        # 确定每一帧新出现的请求个数 确定每个请求出现的时间步
        assert self.env_args["n_requested_frame"] <= self.env_args["n_frame"]
        self.frame_n_accumulate_requests = torch.randint(
            self.env_args["n_init_requests"], 
            self.env_args["n_init_requests"] + self.env_args["n_norm_requests"] + 1, 
            [self.batch_size, self.env_args["n_requested_frame"] - 1], 
            generator=self.rng, 
            device='cpu'
        ).to(self.device).sort().values if not self.debug else torch.randint(
            self.env_args["n_init_requests"], 
            self.env_args["n_init_requests"] + self.env_args["n_norm_requests"] + 1, 
            [1, self.env_args["n_requested_frame"] - 1], 
            generator=self.rng, 
            device='cpu'
        ).to(self.device).repeat(self.batch_size, 1).sort().values
        
        self.frame_n_accumulate_requests = torch.cat((
            torch.tensor([self.env_args["n_init_requests"]], device=self.device).expand(self.batch_size, 1), 
            self.frame_n_accumulate_requests, 
            torch.full([self.batch_size, 
                        self.env_args["n_frame"] - self.env_args["n_requested_frame"] + 1], 
                       self.n_tot_requests, 
                       device=self.device)
            ), dim=1)
        assert self.frame_n_accumulate_requests.shape[1] == self.env_args["n_frame"] + 1

        self.requests["appear"] = (self.requests_arange[None, :, None] < self.frame_n_accumulate_requests[:, None, :]).to(torch.int64).argmax(-1)
        self.pre_obs, self.pre_unobs = self.get_obs()
        return self.pre_obs
    
    @torch.no_grad()
    def get_obs(self):
        # 单步处理订单数量存在上限
        cur_max_consider_requests = self.max_consider_requests
        visible = self.requests["visible"]
        n_exist_requests = visible.sum(-1)
        n_consider_requests = n_exist_requests.clamp_max(cur_max_consider_requests)

        # TODO 有问题的 可见订单不知道为啥会拍在不可见订单之后
        if self.algorithm in ["mapt"]:
        # if True:
            # [batch_size, request_num] bool 至少一辆车与请求r的from重合，并且该车处于空闲状态，并且该请求没被其他车占用
            requests_meet_courier_from = ((self.couriers["target"][:, None, :] == self.requests["from"][:, :, None]) & (self.couriers["time_left"] == 0)[:, None, :]).any(-1) & (self._global["courier_request"] == 0).all(-2)
            requests_meet_courier_station2 = ((self.couriers["target"][:, None, :] == self.requests["station2"][:, :, None]) & (self.couriers["time_left"] == 0)[:, None, :]).any(-1) & (self._global["drone_request"] == 4).all(-2)
            requests_meet_drone_station1 = ((self.drones["target"][:, None, :] == self.requests["station1"][:, :, None]) & (self.drones["time_left"] == 0)[:, None, :]).any(-1) & (self._global["courier_request"] == 3).all(-2)
            # 处于to_pickup或者to_delivery状态，说明请求正在被某辆车处理
            # [batch_size, request_num] bool 是否存在一辆空闲车辆接手了请求
            requests_on_vehicle = (((self._global["courier_request"] == 1) | (self._global["courier_request"] == 2)) & (self.couriers["time_left"] == 0)[:, :, None]).any(-2)
            _index = ((visible * 4 + requests_on_vehicle * 2 + requests_meet_courier_from * 1 + requests_meet_courier_station2 * 1 + requests_meet_drone_station1 * 2) * self.n_tot_requests + self.n_tot_requests - self.requests_arange).argsort(-1, descending=True)
            # 生成请求优先级排序索引 [B, M]
            extractor = _index[:, :cur_max_consider_requests]
        else:
            extractor = torch.zeros(self.batch_size, cur_max_consider_requests, dtype=torch.int64, device=self.device)
            for i in range(self.batch_size):
                _index = torch.where(visible[i])[0][:cur_max_consider_requests]
                extractor[i, :_index.shape[0]] = _index
        
         # 只保留了对于前M条请求的对应部分
        courier_request = self._global["courier_request"].gather(2, extractor[:, None, :].expand(-1, self.n_courier, -1))
        drone_request = self._global["drone_request"].gather(2, extractor[:, None, :].expand(-1, self.n_drone, -1))

        unpadded_requests_mask = torch.arange(cur_max_consider_requests, device=self.device) < n_consider_requests[:, None]
        courier_request.masked_fill_(~unpadded_requests_mask[:, None, :], RELATION.padded.value)
        drone_request.masked_fill_(~unpadded_requests_mask[:, None, :], RELATION.padded.value)

        # 整改
        n_requests_from = torch.stack([torch.bincount(t1[t2], minlength=self.n_node) for t1, t2 in zip(self.requests["from"], visible)])
        n_requests_to = torch.stack([torch.bincount(t1[t2], minlength=self.n_node) for t1, t2 in zip(self.requests["to"], visible)])
        
        n_requests_station1 =  torch.stack([torch.bincount(t1[t2 & (t1 < self.n_node)], minlength=self.n_node) for t1, t2 in zip(self.requests["station1"], visible)])
        n_requests_station2 =  torch.stack([torch.bincount(t1[t2 & (t1 < self.n_node)], minlength=self.n_node) for t1, t2 in zip(self.requests["station2"], visible)])
        
        n_requests_from[self.station_mask] = n_requests_station1[self.station_mask]
        n_requests_to[self.station_mask] = n_requests_station2[self.station_mask]    
        
        courier_has_eq3 = (self._global["courier_request"] == 3).any(dim=1)  # [B, n_request]
        drone_has_eq4   = (self._global["drone_request"] == 4).any(dim=1)    # [B, n_request]
        courier_has_eq5 = (self._global["courier_request"] == 5).any(dim=1)  # [B, n_request]

        visible = self.requests["visible"].bool()  # 确保是 bool

        stage1_mask = visible & (~courier_has_eq3)                      # 所有 courier 与该请求均 != 3
        stage2_mask = visible & courier_has_eq3 & (~drone_has_eq4)      # 存在 courier == 3，且所有 drone 与该请求均 != 4
        stage3_mask = visible & drone_has_eq4 & (~courier_has_eq5)      # 存在 drone == 4，且所有 courier 与该请求均 != 5

        # 每个 batch 的计数（LongTensor: [B]）
        n_stage1 = stage1_mask.sum(dim=1)
        n_stage2 = stage2_mask.sum(dim=1)
        n_stage3 = stage3_mask.sum(dim=1)
        
        obs = {
            "nodes": {
                "coord": self.nodes["coord"],
                # 每个节点作为courier target的数量
                "n_courier": torch.stack([torch.bincount(t1, minlength=self.n_node) for t1 in self.couriers["target"]]),
                # 每个节点作为Drone target的数量
                "n_drone": torch.stack([torch.bincount(t1, minlength=self.n_node) for t1 in self.drones["target"]]),
                # TODO 这里是否可以优化
                "n_requests_from": n_requests_from,
                "n_requests_to": n_requests_to,
            },
            "drones": {
                "target": self.drones["target"].clone(), # inplace update
                "space": self.drones["space"].clone(), # inplace update
                "time_left": self.drones["time_left"].clone(), # inplace update
            },
            "couriers": {
                "target": self.couriers["target"].clone(), # inplace update
                "capacity": self.couriers["capacity"],
                "space": self.couriers["space"].clone(), # inplace update
                "time_left": self.couriers["time_left"].clone(), # inplace update
                # agent don't need to observation cost and value.
            },
            "requests": {
                "from": self.requests["from"].gather(-1, extractor),
                "to": self.requests["to"].gather(-1, extractor),
                "station1": self.requests["station1"].gather(-1, extractor),
                "station2": self.requests["station2"].gather(-1, extractor),
                "value": self.requests["value"].gather(-1, extractor),
                "volumn": self.requests["volumn"].gather(-1, extractor),
                "appear": self.requests["appear"].gather(-1, extractor), # should not be used by only algorithms.
            },
            "global": {
                "frame": torch.full([self.batch_size], self._global["frame"], device=self.device), # int
                "n_exist_requests": n_exist_requests,
                "n_stage1": n_stage1,
                "n_stage2": n_stage2,
                "n_stage3": n_stage3,
                "n_consider_requests": n_consider_requests,
                "node_node": self._global["node_node"],
                "station_idx": self._global["station_idx"],
                "station_mask": self._global["station_mask"],
                "station_node_node": self._global["station_node_node"],
                "courier_request": courier_request.clone(), # inplace update
                "drone_request": drone_request.clone(),
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
        infos = {}
        extractor = self.pre_unobs["extractor"]
        # Stage 1
        # 提取对于第一阶段的request获得指派的两个station
        r_station1_assign = actions["station1"]
        r_station2_assign = actions["station2"]
        r_station1_assign_mask = r_station1_assign != self.n_node
        r_station2_assign_mask = r_station2_assign != self.n_node
        
        # 生成索引
        batch_idx = self.batch_arange[:, None].masked_select(r_station1_assign_mask)
        request_idx = extractor[r_station1_assign_mask]
        self.requests["station1"][batch_idx, request_idx] = r_station1_assign[r_station1_assign_mask]
        
        batch_idx = self.batch_arange[:, None].masked_select(r_station2_assign_mask)
        request_idx = extractor[r_station2_assign_mask]
        self.requests["station2"][batch_idx, request_idx] = r_station2_assign[r_station2_assign_mask]
        
        # action指派courier、drone
        if self.algorithm == "mapdp":
            # Courier Preassign
            courier_preassign = actions.get("courier_preassign", None)
            if courier_preassign is not None:
                assert courier_preassign.shape == extractor.shape
                c_pre_mask = courier_preassign != self.n_courier
                
                c_pre_batch = self.batch_arange[:, None].masked_select(c_pre_mask)
                c_pre_req = extractor[c_pre_mask]
                c_pre_veh = courier_preassign[c_pre_mask]
                
                current_status_target = self._global["courier_request"][c_pre_batch, c_pre_veh, c_pre_req]
                assert (current_status_target == 0).all(), "Can only preassign to available (no_relation) vehicle slots."
                
                current_status_all = self._global["courier_request"][c_pre_batch, :, c_pre_req]
                assert ((current_status_all == 1) | (current_status_all == 2)).sum(dim=-1).eq(0).all(), "Request already assigned/active."

                self._global["courier_request"][c_pre_batch, c_pre_veh, c_pre_req] = RELATION.to_pickup.value

            # Drone Preassign
            drone_preassign = actions.get("drone_preassign", None)
            if drone_preassign is not None:
                assert drone_preassign.shape == extractor.shape
                d_pre_mask = drone_preassign != self.n_drone
                
                d_pre_batch = self.batch_arange[:, None].masked_select(d_pre_mask)
                d_pre_req = extractor[d_pre_mask]
                d_pre_veh = drone_preassign[d_pre_mask]
                
                current_status_target = self._global["drone_request"][d_pre_batch, d_pre_veh, d_pre_req]
                assert (current_status_target == 0).all(), "Can only preassign to available (no_relation) vehicle slots."
                
                current_status_all = self._global["drone_request"][d_pre_batch, :, d_pre_req]
                assert ((current_status_all == 1) | (current_status_all == 2)).sum(dim=-1).eq(0).all(), "Request already assigned/active."

                self._global["drone_request"][d_pre_batch, d_pre_veh, d_pre_req] = RELATION.to_pickup.value

        r_courier_assign = actions["request_courier"]
        r_drone_assign = actions["request_drone"]
        r_courier_assign_mask = r_courier_assign != self.n_courier
        r_drone_assign_mask = r_drone_assign != self.n_drone

        # 改变订单与courier指派关系
        batch_idx = self.batch_arange[:, None].masked_select(r_courier_assign_mask)
        request_idx = extractor[r_courier_assign_mask]
        courier_idx = r_courier_assign[r_courier_assign_mask]

        assert (self.couriers["time_left"][batch_idx, courier_idx] == 0).all()
        assert ((self.couriers["target"][batch_idx, courier_idx] == self.requests["from"][batch_idx, request_idx]) | (self.couriers["target"][batch_idx, courier_idx] == self.requests["station2"][batch_idx, request_idx])).all()
        # if not (self._global["courier_request"][batch_idx, :, request_idx]==0).all():
        #     print(self._global["courier_request"][batch_idx, :, request_idx])
        #     print(self._global["courier_request"][batch_idx, :, request_idx].shape)
        # assert (self._global["courier_request"][batch_idx, :, request_idx]==0).all(), "double check, only `no_relation` requests should be assigned."
        # 检查条件：要不所有车辆和这个请求关系为0，要不只有一辆车与该订单关系为3
        current_relations = self._global["courier_request"][batch_idx, :, request_idx]
        
        if self.algorithm == "mapdp":
            # 允许 preassign (1) 的存在
            # 检查是否有 preassign
            is_preassigned = (current_relations == 1).any(dim=-1)
            if is_preassigned.any():
                # 如果有 preassign，必须全部分配给 preassign 的车辆
                pre_veh_indices = (current_relations[is_preassigned] == 1).max(dim=1).indices
                assign_veh_indices = courier_idx[is_preassigned]
                assert (pre_veh_indices == assign_veh_indices).all(), "Courier assignment conflicts with preassign"
            
            # 合法状态检查:
            # 0: no_relation
            # 1: to_pickup (preassign)
            # 3: deliveryed_stage1
            has_invalid_values = ((current_relations != 0) & (current_relations != 1) & (current_relations != 3)).any(dim=-1)
            assert not has_invalid_values.any(), "Invalid relation values found in assignment"
            
            # 1 的数量最多 1 个
            count_ones = (current_relations == 1).sum(dim=-1)
            assert (count_ones <= 1).all(), "Multiple preassigns found"
            
            # 3 的数量最多 1 个
            count_threes = (current_relations == 3).sum(dim=-1)
            assert (count_threes <= 1).all(), "Multiple stage1_deliveries found"
        else:
            condition_all_zero = (current_relations == 0).all(dim=-1)
            condition_one_three = (current_relations == 3).sum(dim=-1) == 1
            
            if not (condition_all_zero | condition_one_three).all():
                print(current_relations[~(condition_all_zero | condition_one_three)])
                assert False, "Condition failed: Requests must have either all 0 relations or exactly one relation 3."
        
        self._global["courier_request"][batch_idx, courier_idx, request_idx] = RELATION.to_delivery.value
        if not ((self._global["courier_request"] == RELATION.to_delivery.value).sum(-1) <= self.couriers["capacity"]).all():
            print((self._global["courier_request"] == RELATION.to_delivery.value).sum(-1))
        assert ((self._global["courier_request"] == RELATION.to_delivery.value).sum(-1) <= self.couriers["capacity"]).all(), "courier capacity exceeded."

        # 改变订单与drone指派关系
        batch_idx = self.batch_arange[:, None].masked_select(r_drone_assign_mask)
        request_idx = extractor[r_drone_assign_mask]
        drone_idx = r_drone_assign[r_drone_assign_mask]
        
        assert (self.drones["time_left"][batch_idx, drone_idx] == 0).all()
        assert (self.drones["target"][batch_idx, drone_idx] == self.requests["station1"][batch_idx, request_idx]).all()
        current_relations_drone = self._global["drone_request"][batch_idx, :, request_idx]
        # if self.algorithm == "mapdp":
        if True:
            # Check preassign match
            is_preassigned = (current_relations_drone == 1).any(dim=-1)
            if is_preassigned.any():
                pre_veh_indices = (current_relations_drone[is_preassigned] == 1).max(dim=1).indices
                assign_veh_indices = drone_idx[is_preassigned]
                assert (pre_veh_indices == assign_veh_indices).all(), "Drone assignment conflicts with preassign"
            # Allow 0 or 1.
            assert ((current_relations_drone == 0) | (current_relations_drone == 1)).all(), "Drone request relation must be 0 or 1"
            assert ((current_relations_drone == 1).sum(dim=-1) <= 1).all(), "Multiple preassigns found"
        else:
            if not (current_relations_drone==0).all():
                print(current_relations_drone)
                print(current_relations_drone.shape)
                print(request_idx)
            assert (current_relations_drone==0).all(), "double check, only `no_relation` requests should be assigned."
        
        self._global["drone_request"][batch_idx, drone_idx, request_idx] = RELATION.to_delivery.value
        
        if not ((self._global["drone_request"] == RELATION.to_delivery.value).sum(-1) <= 1).all():
            print((self._global["drone_request"] == RELATION.to_delivery.value).sum(-1))
            import pdb
            pdb.set_trace()
        assert ((self._global["drone_request"] == RELATION.to_delivery.value).sum(-1) <= 1).all(), "drone capacity exceeded."
        
        # 解码对于courier和drone的动作指派
        courier_to = actions["courier"]
        drone_to = actions["drone"]
        
        # 已经上路的vehicle不可以被打断
        assert ((self.couriers["time_left"] != 0) <= (courier_to == self.couriers["target"])).all()
        assert ((self.drones["time_left"] != 0) <= (drone_to == self.drones["target"])).all()

        c_dispatch_mask = self.couriers["time_left"] == 0
        d_dispatch_mask = self.drones["time_left"] == 0

        # 更新timeleft
        self.couriers["time_left"][c_dispatch_mask] = self._global["node_node"][self.batch_arange[:, None].masked_select(c_dispatch_mask), self.couriers["target"][c_dispatch_mask], courier_to[c_dispatch_mask]] 
        self.couriers["target"][c_dispatch_mask] = courier_to[c_dispatch_mask]

        # 设置一个值来统计这一个step会增加多少运行dist
        add_dist_courier = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        batch_ids = self.batch_arange[:, None].masked_select(c_dispatch_mask)
        add_dist_courier.index_add_(0, batch_ids, self.couriers["time_left"][c_dispatch_mask])
        
        times = self._global["node_node"][self.batch_arange[:, None].masked_select(d_dispatch_mask), self.drones["target"][d_dispatch_mask], drone_to[d_dispatch_mask]] / self.drone_speed_ratio
        # 用于准确计算drone的cost
        self.drones["float_time_left"][d_dispatch_mask] = times
        self.drones["time_left"][d_dispatch_mask] = times.ceil().to(self.drones["time_left"].dtype)
        self.drones["target"][d_dispatch_mask] = drone_to[d_dispatch_mask]
        
        # 设置一个值来统计这一个step会增加多少运行dist
        add_dist_drone = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        batch_ids = self.batch_arange[:, None].masked_select(d_dispatch_mask)
        add_dist_drone.index_add_(0, batch_ids, self.drones["time_left"][d_dispatch_mask])

        # 正在运行的代价
        # courier每个时间步走1单位距离，所以 (time_left != 0) 就是当前时间步的成本
        current_cost_courier = (self.couriers["time_left"] != 0).sum(-1)
        # drone每个时间步走drone_speed_ratio单位距离（如果float_time_left >= 1）或float_time_left * drone_speed_ratio（如果float_time_left < 1）
        # 当前时间步drone实际移动的距离 = min(1.0, float_time_left) * drone_speed_ratio
        drone_step_distance = self.drones["float_time_left"].clamp(0, 1) * self.drone_speed_ratio
        current_cost_drone = drone_step_distance.sum(-1)
        current_cost = current_cost_courier + current_cost_drone
        
        # 每个vehicle的总运行里程
        self.couriers["cost"] += self.couriers["time_left"] != 0
        # drone的成本需要根据实际移动的距离来记录
        self.drones["cost"] = (self.drones["cost"].float() + drone_step_distance).long()

        # 即将送到的courier
        c_delivery_mask = self.couriers["time_left"] == 1
        # 在这步结束后就要送达的订单数
        # stage1
        delivering_stage1 = c_delivery_mask[:, :, None] & (self._global["courier_request"] == RELATION.to_delivery.value) & \
            (self.couriers["target"][:, :, None] == self.requests["station1"][:, None, :]) & \
            (self._global["courier_request"] != RELATION.deliveryed_stage1.value).all(-2, keepdim=True)# [B,V,R]
        # print(self._global["frame"],delivering_stage1[0][0])
        delivering_value_stage1 = delivering_stage1 * self.requests["value"][:, None, :]
        current_value_stage1 = delivering_value_stage1.sum(dim=-1).sum(dim=-1) # [B,V,R] -> [B,V] -> [B]
        self.couriers["value"] += current_value_stage1.sum(dim=-1).long() # 每个courier的累积价值
        # self.couriers["value_count"] += delivering_stage1.sum(dim=-1) # +每个courier在本次送单中完成的订单数量
        self.couriers_stage1_value_count += delivering_stage1.sum(dim=-1)
        self._global["courier_request"][delivering_stage1] = RELATION.deliveryed_stage1.value # 一个 step 就完成 delivery

        # TODO 前提假设是stage1和stage3的配送员不能是同一个人
        # stage3
        # TODO 当前设置可能存在第一阶段courier直接把货送到to
        # if self.algorithm == "sa" or self.algorithm == "ga":
        #     delivering_stage3 = c_delivery_mask[:, :, None] & (self._global["courier_request"] == RELATION.to_delivery.value) & \
        #     (self.couriers["target"][:, :, None] == self.requests["to"][:, None, :]) & (self._global["courier_request"] == RELATION.deliveryed_stage1.value).any(dim=1, keepdim=True)  # [B,V,R]
        # else:
        delivering_stage3 = c_delivery_mask[:, :, None] & (self._global["courier_request"] == RELATION.to_delivery.value) & \
        (self.couriers["target"][:, :, None] == self.requests["to"][:, None, :])# [B,V,R]
        
        delivering_value_stage3 = delivering_stage3 * self.requests["value"][:, None, :]
        current_value_stage3 = delivering_value_stage3.sum(dim=-1).sum(dim=-1) # [B,V,R] -> [B,V] -> [B]
        self.couriers["value"] += current_value_stage3.sum(dim=-1).long()
        self.couriers["value_count"] += delivering_stage3.sum(dim=-1)
        self._global["courier_request"][delivering_stage3] = RELATION.deliveryed_stage3.value # 真正完成订单

        # 即将送到的drone
        d_delivery_mask = self.drones["time_left"] == 1
        # stage2
        # TODO 正确的基础是to_delivery的drone都会直接飞往目的地station
        delivering_stage2 = d_delivery_mask[:, :, None] & (self._global["drone_request"] == RELATION.to_delivery.value)# [B,V,R]
        delivering_value_stage2 = delivering_stage2 * self.requests["value"][:, None, :]
        current_value_stage2 = delivering_value_stage2.sum(dim=-1).sum(dim=-1) # [B,V,R] -> [B,V] -> [B]
        self.drones["value"] += (current_value_stage2.sum(dim=-1)).long()
        self.drones["value_count"] += delivering_stage2.sum(dim=-1)
        self._global["drone_request"][delivering_stage2] = RELATION.deliveryed_stage2.value # 一个 step 就完成 pickup delivery

        # time_left - 1        
        self.couriers["time_left"] = (self.couriers["time_left"] - 1).clamp_min(0)
        self.drones["time_left"] = (self.drones["time_left"] - 1).clamp_min(0)
        # 用于准确计算drone的cost
        self.drones["float_time_left"] = (self.drones["float_time_left"] - 1).clamp_min(0)

        # 不能超载
        self.couriers["space"] = (self.couriers["capacity"] - ((self._global["courier_request"]==RELATION.to_delivery.value)*self.requests["volumn"][:, None, :]).sum(-1))
        assert (self.couriers["space"] >= 0).all()
        
        self.drones["space"] = (1 - ((self._global["drone_request"]==RELATION.to_delivery.value)*self.requests["volumn"][:, None, :]).sum(-1))
        assert (self.drones["space"] >= 0).all()

        # 更新时间步
        self._global["frame"] += 1

        # 移除已完成的请求，并且更新新的可见请求
        new_request_l = self.frame_n_accumulate_requests[self.batch_arange, self._global["frame"] - 1]
        new_request_r = self.frame_n_accumulate_requests[self.batch_arange, self._global["frame"]]
        self.requests["visible"][self.requests_arange < new_request_l[:, None]] ^= True
        self.requests["visible"][self.requests_arange < new_request_r[:, None]] ^= True
        if not self.deliveryed_visible:
            self.requests["visible"][(self._global["courier_request"]==RELATION.deliveryed_stage3.value).any(dim=-2)] = False
        
        if self.algorithm in trans_func_map:
            trans_func_map[self.algorithm](self, actions)

        obs, unobs = self.get_obs()

        # 得到reward
        # rewards = current_value_stage3 - current_cost_courier * self.courier_cost_ratio - current_cost_drone * self.drone_cost_ratio
        rewards = current_value_stage1 + 5 * current_value_stage2 +  10 * current_value_stage3 - current_cost_courier * self.courier_cost_ratio - current_cost_drone * self.drone_cost_ratio
        # rewards =  - current_cost_courier * self.courier_cost_ratio - current_cost_drone * self.drone_cost_ratio
        # rewards = 0.1 * current_value_stage1 + 0.4 * current_value_stage2 +  current_value_stage3 - current_cost_courier * self.courier_cost_ratio - current_cost_drone * self.drone_cost_ratio
        # 检查batch中是否某个环境已经结束
        dones = torch.full([self.batch_size], self._global["frame"], device=self.device) == self.env_args["n_frame"]

        self.pre_obs = obs
        self.pre_unobs = unobs
        
        infos["full_ratio"] = self.couriers["space"].sum(-1) / self.n_courier
        
        infos["current_finish_ratio"] = self.couriers["value_count"].sum(-1) / self.frame_n_accumulate_requests[self.batch_arange, self._global["frame"]]
        infos["global_finish_ratio"] = self.couriers["value_count"].sum(-1) / self.frame_n_accumulate_requests[:, -1]
        infos["global_finish_stage1_ratio"] = self.couriers_stage1_value_count.sum(-1) / self.frame_n_accumulate_requests[:, -1]
        infos["global_finish_stage2_ratio"] = self.drones["value_count"].sum(-1) / self.frame_n_accumulate_requests[:, -1]
        
        infos["stage1_rewards"] = current_value_stage1
        infos["stage2_rewards"] = current_value_stage2
        infos["stage3_rewards"] = current_value_stage3 - current_cost_courier * self.courier_cost_ratio - current_cost_drone * self.drone_cost_ratio
        infos["punishment"] = current_cost_courier * self.courier_cost_ratio
        infos["courier_cost"] = current_cost_courier * self.courier_cost_ratio
        infos["drone_cost"] = current_cost_drone * self.drone_cost_ratio
        infos["profit"] = current_value_stage3
        return obs, rewards, dones, infos
    
    def seed(self, seed, seed_env):
        self.rng.manual_seed(seed)
        self.rng_env.manual_seed(seed_env)
