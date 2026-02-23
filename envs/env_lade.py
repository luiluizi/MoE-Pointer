import sys
import torch
import numpy as np
import math
from pathlib import Path
from .env import DroneTransferEnv, RELATION

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
occupancy_logged = False
obs_func_map = {}
trans_func_map = {}

CITY_CONFIGS = {
    "hz": {
        "lng_range": [120.079320, 120.203163],
        "lat_range": [30.243830, 30.350527],
        "pickup_csv": "pickup/pickup_hz.csv",
        "delivery_csv": "delivery/delivery_hz.csv",
        "n_station": 6,
        "station_id": [38, 12, 10, 21, 33, 32]
    },
    "cq": {
        "lng_range": [106.422522, 106.546171],
        "lat_range": [29.453564, 29.560914],
        "pickup_csv": "pickup/pickup_cq.csv",
        "delivery_csv": "delivery/delivery_cq.csv",
        "n_station": 6,
        "station_id": [10, 23, 29, 34, 17, 21],
    },
    "sh": {
        "lng_range": [121.424605, 121.550168],
        "lat_range": [31.195720, 31.299977],
        "pickup_csv": "pickup/pickup_sh.csv",
        "delivery_csv": "delivery/delivery_sh.csv",
        "n_station": 6,
        "station_id": [28, 1, 16, 21, 2, 37],
    }
}

def meters_to_degrees(grid_m, at_lat):
    """
    Convert meters to approximate degrees of longitude/latitude at a given latitude (at_lat, in degrees).
    Approximation: 1 deg latitude ≈ 111320 m
                   1 deg longitude ≈ 111320 * cos(lat_rad) m
    Returns (delta_lng_deg, delta_lat_deg)
    """
    lat_rad = math.radians(at_lat)
    deg_per_m_lat = 1.0 / 111320.0  # deg per meter for latitude
    deg_per_m_lng = 1.0 / (111320.0 * math.cos(lat_rad))
    delta_lat = grid_m * deg_per_m_lat
    delta_lng = grid_m * deg_per_m_lng
    return delta_lng, delta_lat

def build_grid(lng_min, lng_max, lat_min, lat_max, delta_lng, delta_lat):
    """
    Construct grid indices and the longitude/latitude of grid centers, returns:
      - grid_lng_edges, grid_lat_edges (arrays containing boundaries)
      - ncols, nrows
    """
    # Ensure the upper-right boundary is included: use np.arange up to beyond max then truncate
    lng_edges = np.arange(lng_min, lng_max + delta_lng*0.5, delta_lng)
    lat_edges = np.arange(lat_min, lat_max + delta_lat*0.5, delta_lat)
    # Append the last boundary if it is smaller than max
    if len(lng_edges) > 0 and lng_edges[-1] < lng_max:
        lng_edges = np.append(lng_edges, lng_max)
    if len(lat_edges) > 0 and lat_edges[-1] < lat_max:
        lat_edges = np.append(lat_edges, lat_max)
    ncols = len(lng_edges) - 1
    nrows = len(lat_edges) - 1
    return lng_edges, lat_edges, ncols, nrows

def assign_points_to_grid(df, lng_col, lat_col, lng_edges, lat_edges):
    """
    Assign points to grid cells. Returns:
    - counts matrix (shape nrows x ncols, row 0 = southernmost -> northernmost)
    - (row_idx, col_idx) for each point (-1 indicates the point is outside the grid)
    """
    # Use np.searchsorted to get the index of the interval each point belongs to
    lng_vals = df[lng_col].to_numpy()
    lat_vals = df[lat_col].to_numpy()
    # indices: col = searchsorted(lng_edges, x) - 1
    col_idx = np.searchsorted(lng_edges, lng_vals, side='right') - 1
    row_idx = np.searchsorted(lat_edges, lat_vals, side='right') - 1
    # Mark points that are out of range
    outside_mask = (col_idx < 0) | (col_idx >= (len(lng_edges)-1)) | (row_idx < 0) | (row_idx >= (len(lat_edges)-1))
    # Label out-of-range points as -1
    col_idx[outside_mask] = -1
    row_idx[outside_mask] = -1
    # Count points in each grid cell
    nrows = len(lat_edges) - 1
    ncols = len(lng_edges) - 1
    counts = np.zeros((nrows, ncols), dtype=int)
    for r, c in zip(row_idx, col_idx):
        if r >= 0 and c >= 0:
            counts[r, c] += 1
    # Note: searchsorted returns indices based on edges from south->north and west->east
    return counts, row_idx, col_idx


class DroneTransferEnvLADE(DroneTransferEnv):
    def __init__(self, **kwargs):
        self.env_args = kwargs["env_args"]
        self.deliveryed_visible = self.env_args["deliveryed_visible"]
        self.min_direct_dist = self.env_args['min_direct_dist']
        self.debug = self.env_args["debug"]
        self.algorithm = kwargs["algorithm"]
        
        self.city = kwargs.get("city", "hz")  # hz/cq/sh
        self.is_train = kwargs["is_train"]
        self.train_days = kwargs.get("train_days", 60)
        self.test_days = kwargs.get("test_days", 20)
        
        self.drone_speed_ratio = self.env_args.get('drone_speed_ratio', 4)
        self.drone_cost_ratio = self.env_args['dist_cost_drone']
        self.courier_cost_ratio = self.env_args['dist_cost_courier']
        self.request_profit_ratio = self.env_args['dist_req_profit']
        
        # Set up environment entities
        self.n_node = self.env_args["n_node"]
        self.n_station = self.env_args["n_station"]
        self.n_courier = self.env_args["n_courier"]
        self.n_drone = self.env_args["n_drone"]
        
        self.n_tot_requests = self.env_args["n_init_requests"] + self.env_args["n_norm_requests"]
        self.max_consider_requests = self.n_tot_requests if self.env_args["max_consider_requests"] == -1 else self.env_args["max_consider_requests"]
        
        # Dataset specific parameter
        self.env_args["n_node"] = self.n_node
        self.env_args["n_courier"] = self.n_courier
        self.env_args['n_drone'] = self.n_drone
        self.env_args["n_tot_requests"] = self.n_tot_requests
        
        self.device = kwargs["device"]
        self.rng = torch.Generator(self.device)
        self.rng_env = torch.Generator(self.device)
        
        self.rng.manual_seed(kwargs["seed"])
        self.rng_env.manual_seed(kwargs["seed_env"])
        
        # load dataset
        self.load_dataset()
        
        if self.is_train:
            self.batch_size = kwargs["batch_size"]
        else:
            # Use test set data
            test_from = self.cached.get("test_requests_from", self.cached.get("train_requests_from"))
            self.batch_size = min(test_from.shape[0], kwargs["batch_size"])
        
        self.info_batch_size = self.batch_size if self.env_args["info_batch_size"] == -1 else self.env_args["info_batch_size"]
        
        self.batch_arange = torch.arange(self.batch_size, device=self.device)
        self.node_arange = torch.arange(self.n_node, device=self.device)
        self.requests_arange = torch.arange(self.n_tot_requests, device=self.device)
    
    def generate_filename(self):
        """Generate dataset file names based on parameters"""
        split = "train" if self.is_train else "test"
        return f"{self.city}_{split}_node{self.n_node}_station{self.n_station}_totreq{self.n_tot_requests}_mindist{self.min_direct_dist}.pt"
    
    def load_dataset(self):
        """Load data from pre-generated dataset files"""
        # Construct the file path
        base_dir = project_root
        processed_dir = base_dir / "data" / "LADE" / "processed"
        filename = self.generate_filename()
        file_path = processed_dir / filename
            
        # Check if the file exists
        if not file_path.exists():
            raise FileNotFoundError(
                f"Dataset file does not exist: {file_path}\n"
                f"Please run data/LADE/lade_process.py first to generate the dataset.\n"
                f"Expected file name format: {{city}}_{{split}}_node{{n_node}}_station{{n_station}}_totreq{{n_tot_requests}}_mindist{{min_direct_dist}}.pt"
            )
                
        # load data
        print(f"load dataset: {file_path}")
        data = torch.load(file_path, map_location=self.device, weights_only=False)
        
        # Validate the metadata
        metadata = data.get("metadata", {})
        expected_params = {
            "city": self.city,
            "n_node": self.n_node,
            "n_station": self.n_station,
            "n_tot_requests": self.n_tot_requests,
            "min_direct_dist": self.min_direct_dist,
            "n_frame": self.env_args["n_frame"],
            "n_requested_frame": self.env_args["n_requested_frame"],
            "n_init_requests": self.env_args["n_init_requests"],
            "n_norm_requests": self.env_args["n_norm_requests"],
        }
        
        mismatches = []
        for key, expected_value in expected_params.items():
            if key in metadata:
                actual_value = metadata[key]
                if actual_value != expected_value:
                    mismatches.append(f"{key}: Expected {expected_value}, Actual {actual_value}")
        
        if mismatches:
            raise ValueError(
                f"Mismatch in dataset metadata:\n" + "\n".join(mismatches) + 
                f"\nFile path: {file_path}"
            )
        
        # Load the data into the cache
        self.cached = {}
        self.cached["coord"] = data["coord"].to(self.device)
        self.cached["node_node"] = data["node_node"].to(self.device)
        self.cached["station_idx"] = data["station_idx"].to(self.device)
        self.cached["station_mask"] = data["station_mask"].to(self.device)
        self.cached["station_node_node"] = data["station_node_node"].to(self.device)
        
        # Select training set or test set data based on is_train
        if self.is_train:
            self.cached["train_requests_from"] = data["requests_from"].to(self.device)
            self.cached["train_requests_to"] = data["requests_to"].to(self.device)
            self.cached["train_requests_value"] = data["requests_value"].to(self.device)
            self.cached["train_frame_accumulate"] = data["frame_accumulate"].to(self.device)
        else:
            self.cached["test_requests_from"] = data["requests_from"].to(self.device)
            self.cached["test_requests_to"] = data["requests_to"].to(self.device)
            self.cached["test_requests_value"] = data["requests_value"].to(self.device)
            self.cached["test_frame_accumulate"] = data["frame_accumulate"].to(self.device)
        
        print(f"Dataset loaded successfully! Number of instances: {metadata.get('n_instances', 'unknown')}")
    
    def reset(self, **kwargs):
        """Reset the environment and select training/test set data based on is_train"""
        if self.debug:
            self.rng.manual_seed(123321)
            print("use debug seed 123321.")
        
        # Select the dataset
        if self.is_train:
            requests_from = self.cached["train_requests_from"]
            requests_to = self.cached["train_requests_to"]
            requests_value = self.cached["train_requests_value"]
            frame_accumulate = self.cached["train_frame_accumulate"]
        else:
            requests_from = self.cached["test_requests_from"]
            requests_to = self.cached["test_requests_to"]
            requests_value = self.cached["test_requests_value"]
            frame_accumulate = self.cached["test_frame_accumulate"]
        
        # Randomly select an instance
        n_instances = requests_from.shape[0]
        if self.is_train:
            batch_idx = torch.randint(0, n_instances, (self.batch_size,), generator=self.rng, device=self.device)
        else:
            rng_test = torch.Generator(self.device)
            rng_test.manual_seed(0)
            batch_idx = torch.randperm(n_instances, generator=rng_test, device=self.device)[:self.batch_size]
        
        # Set the node coordinates
        self.nodes = {
            "coord": self.cached["coord"].repeat(self.batch_size, 1, 1),
        }
        
        # Set the station information
        station_idx = self.cached["station_idx"].unsqueeze(0).repeat(self.batch_size, 1)
        self.station_idx = station_idx
        self.station_mask = self.cached["station_mask"].unsqueeze(0).repeat(self.batch_size, 1)
        self.station_node_node = self.cached["station_node_node"].unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        # Initialize the couriers
        self.couriers = {
            "target": torch.randint(0, self.n_node, [self.batch_size, self.n_courier], generator=self.rng, device=self.device),
            "capacity": torch.full([self.batch_size, self.n_courier], self.env_args["max_capacity"], device=self.device),
            "space": torch.full([self.batch_size, self.n_courier], self.env_args["max_capacity"], device=self.device),
            "time_left": torch.full([self.batch_size, self.n_courier], 0, device=self.device),
            "cost": torch.zeros([self.batch_size, self.n_courier], dtype=torch.int64, device=self.device),
            "value": torch.zeros([self.batch_size, self.n_courier], dtype=torch.int64, device=self.device),
            "value_count": torch.zeros([self.batch_size, self.n_courier], dtype=torch.int64, device=self.device),
        }
        self.couriers_stage1_value_count = torch.zeros([self.batch_size, self.n_courier], dtype=torch.int64, device=self.device)
        
        # Initialize the drones
        if not self.debug:
            cycle_indices = torch.arange(self.n_drone, device=self.device) % self.n_station
            batch_offsets = torch.randint(0, self.n_station, (self.batch_size,), generator=self.rng, device=self.device)
            rand_idx = (cycle_indices.unsqueeze(0) + batch_offsets.unsqueeze(1)) % self.n_station
            drones_target = torch.gather(
                station_idx[:,:-1].unsqueeze(1).expand(-1, self.n_drone, -1),
                2,
                rand_idx.unsqueeze(2)
            ).squeeze(2)
        else:
            rand_idx = torch.randint(0, self.n_station, (1, self.n_drone), generator=self.rng, device=self.device)
            drones_target = torch.gather(
                station_idx[:1].unsqueeze(1).expand(-1, self.n_drone, -1),
                2,
                rand_idx.unsqueeze(2)
            ).squeeze(2).repeat(self.batch_size, 1)
        
        self.drones = {
            "target": drones_target,
            "space": torch.full([self.batch_size, self.n_drone], 1, device=self.device),
            "time_left": torch.full([self.batch_size, self.n_drone], 0, device=self.device),
            "float_time_left": torch.full([self.batch_size, self.n_drone], 0.0, device=self.device),
            "cost": torch.zeros([self.batch_size, self.n_drone], dtype=torch.int64, device=self.device),
            "value": torch.zeros([self.batch_size, self.n_drone], dtype=torch.int64, device=self.device),
            "value_count": torch.zeros([self.batch_size, self.n_drone], dtype=torch.int64, device=self.device),
        }
        
        visible = torch.full([self.batch_size, self.n_tot_requests], False, device=self.device)
        initial_counts = frame_accumulate[batch_idx, 0]  # Number of orders in the first time step
        visible[self.requests_arange[None, :] < initial_counts[:, None]] = True
        
        self.requests = {
            "from": requests_from[batch_idx],
            "to": requests_to[batch_idx],
            "value": requests_value[batch_idx],
            "volumn": torch.ones(self.batch_size, self.n_tot_requests, dtype=torch.int64, device=self.device),
            "visible": visible,
            "appear": (self.requests_arange[None, :, None] < frame_accumulate[batch_idx, None, :]).to(torch.int64).argmax(-1),
            "station1": torch.full([self.batch_size, self.n_tot_requests], self.n_node, device=self.device),
            "station2": torch.full([self.batch_size, self.n_tot_requests], self.n_node, device=self.device),
        }
        
        # Set the relationship matrix
        rel_mat_courier = torch.full([self.batch_size, self.n_courier, self.n_tot_requests], RELATION.no_relation.value, device=self.device)
        rel_mat_courier.masked_fill_(
            (self.requests_arange >= frame_accumulate[batch_idx, -1:])[:, None, :],
            RELATION.padded.value
        )
        
        rel_mat_drone = torch.full([self.batch_size, self.n_drone, self.n_tot_requests], RELATION.no_relation.value, device=self.device)
        rel_mat_drone.masked_fill_(
            (self.requests_arange >= frame_accumulate[batch_idx, -1:])[:, None, :],
            RELATION.padded.value
        )
        
        # Set the global state
        self._global = {
            "frame": 0,
            "n_exist_requests": frame_accumulate[batch_idx, 0],
            "node_node": self.cached["node_node"].repeat(self.batch_size, 1, 1),
            "station_idx": station_idx,
            "station_mask": self.station_mask,
            "station_node_node": self.station_node_node,
            "courier_request": rel_mat_courier,
            "drone_request": rel_mat_drone,
        }
        
        self.node_node = self._global['node_node']
        self.frame_n_accumulate_requests = frame_accumulate[batch_idx]
 
        self.pre_obs, self.pre_unobs = self.get_obs()
        return self.pre_obs
    
    def seed(self, seed, seed_env):
        self.rng.manual_seed(seed)
        self.rng_env.manual_seed(seed_env)

