import math
import torch
from enum import Enum
from .multiagentenv import MultiAgentEnv

occupancy_logged = False

obs_func_map = {}
trans_func_map = {}

class RELATION(Enum):
    no_relation = 0
    to_pickup = 1
    to_delivery = 2
    deliveryed_stage1 = 3
    deliveryed_stage2 = 4
    deliveryed_stage3 = 5
    padded = 6 # Cannot be embedded
    skipped = 7 # Can be decoded but is skipped due to policy settings

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
        self.deliveryed_visible = self.env_args["deliveryed_visible"]
        self.info_batch_size = self.batch_size if self.env_args["info_batch_size"] == -1 else self.env_args["info_batch_size"]
        self.debug = self.env_args["debug"] # all envs use one seed.
        self.algorithm = kwargs["algorithm"]

        self.dist_distribution = self.env_args["dist_distribution"]
        assert self.dist_distribution == "2D", "dist_distribution must be '2D'"
       
        self.dist_cost = self.env_args["dist_cost"]

        # Environment entity setup
        self.n_node = self.env_args["n_node"]
        self.n_station = self.env_args["n_station"]
        self.n_courier = self.env_args["n_courier"]
        self.n_drone = self.env_args["n_drone"] 
        self.min_direct_dist = self.env_args['min_direct_dist'] # 设置订单的最短距离
        self.drone_speed_ratio = self.env_args.get('drone_speed_ratio', 4)
        
        # Setup for vehicle cost and order profit
        self.drone_cost_ratio = self.env_args['dist_cost_drone']
        self.courier_cost_ratio = self.env_args['dist_cost_courier']
        self.request_profit_ratio = self.env_args['dist_req_profit']
        
        self.n_tot_requests = self.env_args["n_init_requests"] + self.env_args["n_norm_requests"]
        self.max_consider_requests = self.n_tot_requests if self.env_args["max_consider_requests"] == -1 else self.env_args["max_consider_requests"]
        self.device = kwargs["device"]
        self.rng = torch.Generator('cpu')
        self.rng_env = torch.Generator('cpu')

        self.batch_arange = torch.arange(self.batch_size, device=self.device)
        self.node_arange = torch.arange(self.n_node, device=self.device)
        self.requests_arange = torch.arange(self.n_tot_requests, device=self.device)

    def generate_non_overlapping(self, num_samples, node_node, min_direct_dist=0):
        non_station_idx = torch.nonzero(~self.station_mask[0], as_tuple=False).reshape(-1)
        n_non_station = non_station_idx.numel()
        assert n_non_station >= 2, "Insufficient number of non-transfer station nodes to generate orders"
        
        non_station_dist = node_node[non_station_idx][:, non_station_idx]
        valid_mask = (non_station_dist >= min_direct_dist) & ~torch.eye(n_non_station, dtype=torch.bool, device=self.device)
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
            # Randomly select n_node nodes
            node_indices = torch.randperm(n_node_tot, generator=self.rng_env, device='cpu')[:n_node].to(self.device)
            node_coord = grid_coord[node_indices]

            # Calculate Manhattan distance between nodes
            node_node = (node_coord[:, None] - node_coord[None, :]).abs().sum(-1)
            node_node = node_node.repeat(self.batch_size, 1, 1)
            node_coord = node_coord.repeat(self.batch_size, 1, 1)
            print("node average dist", torch.mean(node_node.float()).item())

            # Randomly select n_station transfer stations
            station_indices = torch.randperm(n_node, generator=self.rng_env, device='cpu')[:n_station].to(self.device)

            station_idx = torch.cat([station_indices, torch.tensor([n_node], device=self.device)]).unsqueeze(0).repeat(self.batch_size, 1)
            station_mask = torch.zeros((self.batch_size, n_node), dtype=torch.bool, device=self.device)
            station_mask[:, station_indices] = True 

            # Calculate distance between transfer stations
            submat = node_node[0, station_indices][:, station_indices]
            station_node_node = submat.unsqueeze(0).repeat(self.batch_size, 1, 1)
        
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
        # Generate initial drone distribution, avoiding concentration at the same point as much as possible
        if not self.debug:
            cycle_indices = torch.arange(self.n_drone, device=self.device) % self.n_station
            batch_offsets = torch.randint(0, self.n_station, (self.batch_size,), generator=self.rng, device='cpu').to(self.device)

            rand_idx = (cycle_indices.unsqueeze(0) + batch_offsets.unsqueeze(1)) % self.n_station
            drones_target = torch.gather(self.station_idx[:,:-1].unsqueeze(1).expand(-1, self.n_drone, -1), 2, rand_idx.unsqueeze(2)).squeeze(2)
        else:
            rand_idx = torch.randint(0, self.n_station, (1, self.n_drone), generator=self.rng, device='cpu' ).to(self.device)
            drones_target = torch.gather(self.station_idx[:1].unsqueeze(1).expand(-1, self.n_drone, -1), 2, rand_idx.unsqueeze(2)).squeeze(2).repeat(self.batch_size, 1)

        # Set initial state of entities
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
        
        # Determine the number of newly appeared requests per frame and the time step of each request's appearance
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
        # Due to GPU memory limitations, the number of orders processed per step is capped
        cur_max_consider_requests = self.max_consider_requests
        visible = self.requests["visible"]
        n_exist_requests = visible.sum(-1)
        n_consider_requests = n_exist_requests.clamp_max(cur_max_consider_requests)

        if self.algorithm in ["moe_pointer"]:
            # [batch_size, request_num] bool: At least one vehicle overlaps with the "from" location of request r, the vehicle is in an idle state, and the request is not occupied by other vehicles
            requests_meet_courier_from = ((self.couriers["target"][:, None, :] == self.requests["from"][:, :, None]) & (self.couriers["time_left"] == 0)[:, None, :]).any(-1) & (self._global["courier_request"] == 0).all(-2)
            requests_meet_courier_station2 = ((self.couriers["target"][:, None, :] == self.requests["station2"][:, :, None]) & (self.couriers["time_left"] == 0)[:, None, :]).any(-1) & (self._global["drone_request"] == 4).all(-2)
            requests_meet_drone_station1 = ((self.drones["target"][:, None, :] == self.requests["station1"][:, :, None]) & (self.drones["time_left"] == 0)[:, None, :]).any(-1) & (self._global["courier_request"] == 3).all(-2)
            # Being in to_pickup or to_delivery state indicates that the request is being processed by a vehicle
            # [batch_size, request_num] bool: Whether there is an idle vehicle that has taken over the request
            requests_on_vehicle = (((self._global["courier_request"] == 1) | (self._global["courier_request"] == 2)) & (self.couriers["time_left"] == 0)[:, :, None]).any(-2)
            _index = ((visible * 4 + requests_on_vehicle * 2 + requests_meet_courier_from * 1 + requests_meet_courier_station2 * 1 + requests_meet_drone_station1 * 2) * self.n_tot_requests + self.n_tot_requests - self.requests_arange).argsort(-1, descending=True)
            extractor = _index[:, :cur_max_consider_requests]
        else:
            extractor = torch.zeros(self.batch_size, cur_max_consider_requests, dtype=torch.int64, device=self.device)
            for i in range(self.batch_size):
                _index = torch.where(visible[i])[0][:cur_max_consider_requests]
                extractor[i, :_index.shape[0]] = _index
        
        # Only retain the corresponding parts for the first M requests
        courier_request = self._global["courier_request"].gather(2, extractor[:, None, :].expand(-1, self.n_courier, -1))
        drone_request = self._global["drone_request"].gather(2, extractor[:, None, :].expand(-1, self.n_drone, -1))

        unpadded_requests_mask = torch.arange(cur_max_consider_requests, device=self.device) < n_consider_requests[:, None]
        courier_request.masked_fill_(~unpadded_requests_mask[:, None, :], RELATION.padded.value)
        drone_request.masked_fill_(~unpadded_requests_mask[:, None, :], RELATION.padded.value)

        # Calculate statistical properties of nodes
        n_requests_from = torch.stack([torch.bincount(t1[t2], minlength=self.n_node) for t1, t2 in zip(self.requests["from"], visible)])
        n_requests_to = torch.stack([torch.bincount(t1[t2], minlength=self.n_node) for t1, t2 in zip(self.requests["to"], visible)])
        n_requests_station1 =  torch.stack([torch.bincount(t1[t2 & (t1 < self.n_node)], minlength=self.n_node) for t1, t2 in zip(self.requests["station1"], visible)])
        n_requests_station2 =  torch.stack([torch.bincount(t1[t2 & (t1 < self.n_node)], minlength=self.n_node) for t1, t2 in zip(self.requests["station2"], visible)])
        
        n_requests_from[self.station_mask] = n_requests_station1[self.station_mask]
        n_requests_to[self.station_mask] = n_requests_station2[self.station_mask]    
        
        courier_has_eq3 = (self._global["courier_request"] == 3).any(dim=1)  # [B, n_request]
        drone_has_eq4   = (self._global["drone_request"] == 4).any(dim=1)    # [B, n_request]
        courier_has_eq5 = (self._global["courier_request"] == 5).any(dim=1)  # [B, n_request]

        visible = self.requests["visible"].bool() 

        stage1_mask = visible & (~courier_has_eq3)   
        stage2_mask = visible & courier_has_eq3 & (~drone_has_eq4)
        stage3_mask = visible & drone_has_eq4 & (~courier_has_eq5) 

        n_stage1 = stage1_mask.sum(dim=1)
        n_stage2 = stage2_mask.sum(dim=1)
        n_stage3 = stage3_mask.sum(dim=1)
        
        obs = {
            "nodes": {
                "coord": self.nodes["coord"],
                # Number of times each node is used as a courier target
                "n_courier": torch.stack([torch.bincount(t1, minlength=self.n_node) for t1 in self.couriers["target"]]),
                # Number of times each node is used as a drone target
                "n_drone": torch.stack([torch.bincount(t1, minlength=self.n_node) for t1 in self.drones["target"]]),
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
            },
            "requests": {
                "from": self.requests["from"].gather(-1, extractor),
                "to": self.requests["to"].gather(-1, extractor),
                "station1": self.requests["station1"].gather(-1, extractor),
                "station2": self.requests["station2"].gather(-1, extractor),
                "value": self.requests["value"].gather(-1, extractor),
                "volumn": self.requests["volumn"].gather(-1, extractor),
                "appear": self.requests["appear"].gather(-1, extractor), 
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
        # Extract the two stations assigned to the requests
        r_station1_assign = actions["station1"]
        r_station2_assign = actions["station2"]
        r_station1_assign_mask = r_station1_assign != self.n_node
        r_station2_assign_mask = r_station2_assign != self.n_node
        
        # Generate index
        batch_idx = self.batch_arange[:, None].masked_select(r_station1_assign_mask)
        request_idx = extractor[r_station1_assign_mask]
        self.requests["station1"][batch_idx, request_idx] = r_station1_assign[r_station1_assign_mask]
        
        batch_idx = self.batch_arange[:, None].masked_select(r_station2_assign_mask)
        request_idx = extractor[r_station2_assign_mask]
        self.requests["station2"][batch_idx, request_idx] = r_station2_assign[r_station2_assign_mask]
        
        # action assign courier/drone
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

        # update the assignment relationship between requests and couriers
        batch_idx = self.batch_arange[:, None].masked_select(r_courier_assign_mask)
        request_idx = extractor[r_courier_assign_mask]
        courier_idx = r_courier_assign[r_courier_assign_mask]

        assert (self.couriers["time_left"][batch_idx, courier_idx] == 0).all()
        assert ((self.couriers["target"][batch_idx, courier_idx] == self.requests["from"][batch_idx, request_idx]) | (self.couriers["target"][batch_idx, courier_idx] == self.requests["station2"][batch_idx, request_idx])).all()
        current_relations = self._global["courier_request"][batch_idx, :, request_idx]
        
        if self.algorithm == "mapdp":
            # Allow the existence of preassign (1)
            # Check if preassign exists
            is_preassigned = (current_relations == 1).any(dim=-1)
            if is_preassigned.any():
                # If preassign exists, all must be assigned to the preassigned vehicles
                pre_veh_indices = (current_relations[is_preassigned] == 1).max(dim=1).indices
                assign_veh_indices = courier_idx[is_preassigned]
                assert (pre_veh_indices == assign_veh_indices).all(), "Courier assignment conflicts with preassign"
            
            # 0: no_relation
            # 1: to_pickup (preassign)
            # 3: deliveryed_stage1
            has_invalid_values = ((current_relations != 0) & (current_relations != 1) & (current_relations != 3)).any(dim=-1)
            assert not has_invalid_values.any(), "Invalid relation values found in assignment"
            
            # At most one 1 is allowed
            count_ones = (current_relations == 1).sum(dim=-1)
            assert (count_ones <= 1).all(), "Multiple preassigns found"
            
            # At most one 3 is allowed
            count_threes = (current_relations == 3).sum(dim=-1)
            assert (count_threes <= 1).all(), "Multiple stage1_deliveries found"
        else:
            condition_all_zero = (current_relations == 0).all(dim=-1)
            condition_one_three = (current_relations == 3).sum(dim=-1) == 1
            
            if not (condition_all_zero | condition_one_three).all():
                assert False, "Condition failed: Requests must have either all 0 relations or exactly one relation 3."
        
        self._global["courier_request"][batch_idx, courier_idx, request_idx] = RELATION.to_delivery.value
        assert ((self._global["courier_request"] == RELATION.to_delivery.value).sum(-1) <= self.couriers["capacity"]).all(), "courier capacity exceeded."

        # update the assignment relationship between requests and drones
        batch_idx = self.batch_arange[:, None].masked_select(r_drone_assign_mask)
        request_idx = extractor[r_drone_assign_mask]
        drone_idx = r_drone_assign[r_drone_assign_mask]
        
        assert (self.drones["time_left"][batch_idx, drone_idx] == 0).all()
        assert (self.drones["target"][batch_idx, drone_idx] == self.requests["station1"][batch_idx, request_idx]).all()
        current_relations_drone = self._global["drone_request"][batch_idx, :, request_idx]
        if self.algorithm == "mapdp":
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
            assert (current_relations_drone==0).all(), "double check, only `no_relation` requests should be assigned."
        
        self._global["drone_request"][batch_idx, drone_idx, request_idx] = RELATION.to_delivery.value
        
        assert ((self._global["drone_request"] == RELATION.to_delivery.value).sum(-1) <= 1).all(), "drone capacity exceeded."
        
        # Decode action assignments for couriers and drones
        courier_to = actions["courier"]
        drone_to = actions["drone"]
        
        # Vehicles already on route cannot be interrupted
        assert ((self.couriers["time_left"] != 0) <= (courier_to == self.couriers["target"])).all()
        assert ((self.drones["time_left"] != 0) <= (drone_to == self.drones["target"])).all()
        
        c_dispatch_mask = self.couriers["time_left"] == 0
        d_dispatch_mask = self.drones["time_left"] == 0

        # update timeleft
        self.couriers["time_left"][c_dispatch_mask] = self._global["node_node"][self.batch_arange[:, None].masked_select(c_dispatch_mask), self.couriers["target"][c_dispatch_mask], courier_to[c_dispatch_mask]] 
        self.couriers["target"][c_dispatch_mask] = courier_to[c_dispatch_mask]

        # Set a value to count how much running distance (dist) will increase in this step
        add_dist_courier = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        batch_ids = self.batch_arange[:, None].masked_select(c_dispatch_mask)
        add_dist_courier.index_add_(0, batch_ids, self.couriers["time_left"][c_dispatch_mask])
        
        times = self._global["node_node"][self.batch_arange[:, None].masked_select(d_dispatch_mask), self.drones["target"][d_dispatch_mask], drone_to[d_dispatch_mask]] / self.drone_speed_ratio
        # Used to accurately calculate drone cost
        self.drones["float_time_left"][d_dispatch_mask] = times
        self.drones["time_left"][d_dispatch_mask] = times.ceil().to(self.drones["time_left"].dtype)
        self.drones["target"][d_dispatch_mask] = drone_to[d_dispatch_mask]
        
        # Set a value to count how much running distance (dist) will increase in this step
        add_dist_drone = torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        batch_ids = self.batch_arange[:, None].masked_select(d_dispatch_mask)
        add_dist_drone.index_add_(0, batch_ids, self.drones["time_left"][d_dispatch_mask])

        # Cost of ongoing operation
        # A courier travels 1 unit of distance per time step, so (time_left != 0) is the cost for the current time step
        current_cost_courier = (self.couriers["time_left"] != 0).sum(-1)
        # A drone travels drone_speed_ratio units of distance per time step (if float_time_left >= 1) or float_time_left * drone_speed_ratio (if float_time_left < 1)
        # Actual distance traveled by the drone in the current time step = min(1.0, float_time_left) * drone_speed_ratio
        drone_step_distance = self.drones["float_time_left"].clamp(0, 1) * self.drone_speed_ratio
        current_cost_drone = drone_step_distance.sum(-1)
        
        self.couriers["cost"] += self.couriers["time_left"] != 0
        self.drones["cost"] = (self.drones["cost"].float() + drone_step_distance).long()

        # Courier about to complete delivery
        c_delivery_mask = self.couriers["time_left"] == 1
        # stage1
        delivering_stage1 = c_delivery_mask[:, :, None] & (self._global["courier_request"] == RELATION.to_delivery.value) & \
            (self.couriers["target"][:, :, None] == self.requests["station1"][:, None, :]) & \
            (self._global["courier_request"] != RELATION.deliveryed_stage1.value).all(-2, keepdim=True)# [B,V,R]
        delivering_value_stage1 = delivering_stage1 * self.requests["value"][:, None, :]
        current_value_stage1 = delivering_value_stage1.sum(dim=-1).sum(dim=-1) # [B,V,R] -> [B,V] -> [B]
        self.couriers["value"] += current_value_stage1.sum(dim=-1).long() 
        self.couriers_stage1_value_count += delivering_stage1.sum(dim=-1)
        self._global["courier_request"][delivering_stage1] = RELATION.deliveryed_stage1.value # Complete delivery in one step

        # The courier for stage1 and stage3 cannot be the same
        # stage3
        delivering_stage3 = c_delivery_mask[:, :, None] & (self._global["courier_request"] == RELATION.to_delivery.value) & \
        (self.couriers["target"][:, :, None] == self.requests["to"][:, None, :])# [B,V,R]
        
        delivering_value_stage3 = delivering_stage3 * self.requests["value"][:, None, :]
        current_value_stage3 = delivering_value_stage3.sum(dim=-1).sum(dim=-1) # [B,V,R] -> [B,V] -> [B]
        self.couriers["value"] += current_value_stage3.sum(dim=-1).long()
        self.couriers["value_count"] += delivering_stage3.sum(dim=-1)
        self._global["courier_request"][delivering_stage3] = RELATION.deliveryed_stage3.value 

        # Drone about to complete delivery
        d_delivery_mask = self.drones["time_left"] == 1
        # stage2
        # to_delivery的drone都会直接飞往目的地station
        delivering_stage2 = d_delivery_mask[:, :, None] & (self._global["drone_request"] == RELATION.to_delivery.value)# [B,V,R]
        delivering_value_stage2 = delivering_stage2 * self.requests["value"][:, None, :]
        current_value_stage2 = delivering_value_stage2.sum(dim=-1).sum(dim=-1) # [B,V,R] -> [B,V] -> [B]
        self.drones["value"] += (current_value_stage2.sum(dim=-1)).long()
        self.drones["value_count"] += delivering_stage2.sum(dim=-1)
        self._global["drone_request"][delivering_stage2] = RELATION.deliveryed_stage2.value

        # time_left - 1        
        self.couriers["time_left"] = (self.couriers["time_left"] - 1).clamp_min(0)
        self.drones["time_left"] = (self.drones["time_left"] - 1).clamp_min(0)
        self.drones["float_time_left"] = (self.drones["float_time_left"] - 1).clamp_min(0)

        # Overload check
        self.couriers["space"] = (self.couriers["capacity"] - ((self._global["courier_request"]==RELATION.to_delivery.value)*self.requests["volumn"][:, None, :]).sum(-1))
        assert (self.couriers["space"] >= 0).all()
        
        self.drones["space"] = (1 - ((self._global["drone_request"]==RELATION.to_delivery.value)*self.requests["volumn"][:, None, :]).sum(-1))
        assert (self.drones["space"] >= 0).all()

        self._global["frame"] += 1

        # Remove completed requests and update the new visible requests
        new_request_l = self.frame_n_accumulate_requests[self.batch_arange, self._global["frame"] - 1]
        new_request_r = self.frame_n_accumulate_requests[self.batch_arange, self._global["frame"]]
        self.requests["visible"][self.requests_arange < new_request_l[:, None]] ^= True
        self.requests["visible"][self.requests_arange < new_request_r[:, None]] ^= True
        if not self.deliveryed_visible:
            self.requests["visible"][(self._global["courier_request"]==RELATION.deliveryed_stage3.value).any(dim=-2)] = False
        
        if self.algorithm in trans_func_map:
            trans_func_map[self.algorithm](self, actions)

        obs, unobs = self.get_obs()

        rewards = current_value_stage1 + 5 * current_value_stage2 +  10 * current_value_stage3 - current_cost_courier * self.courier_cost_ratio - current_cost_drone * self.drone_cost_ratio
        # Check if any environment in the batch has ended
        dones = torch.full([self.batch_size], self._global["frame"], device=self.device) == self.env_args["n_frame"]

        self.pre_obs = obs
        self.pre_unobs = unobs

        infos["global_finish_ratio"] = self.couriers["value_count"].sum(-1) / self.frame_n_accumulate_requests[:, -1]
        infos["global_finish_stage1_ratio"] = self.couriers_stage1_value_count.sum(-1) / self.frame_n_accumulate_requests[:, -1]
        infos["global_finish_stage2_ratio"] = self.drones["value_count"].sum(-1) / self.frame_n_accumulate_requests[:, -1]
        
        infos["stage1_rewards"] = current_value_stage1
        infos["stage2_rewards"] = current_value_stage2
        infos["stage3_rewards"] = current_value_stage3 - current_cost_courier * self.courier_cost_ratio - current_cost_drone * self.drone_cost_ratio
        infos["courier_cost"] = current_cost_courier * self.courier_cost_ratio
        infos["drone_cost"] = current_cost_drone * self.drone_cost_ratio
        infos["profit"] = current_value_stage3
        return obs, rewards, dones, infos
    
    def seed(self, seed, seed_env):
        self.rng.manual_seed(seed)
        self.rng_env.manual_seed(seed_env)
