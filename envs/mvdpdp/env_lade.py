import torch
import numpy as np
import math
from pathlib import Path

from .env import DroneTransferEnv, RELATION

occupancy_logged = False
obs_func_map = {}
trans_func_map = {}

# 城市配置
CITY_CONFIGS = {
    "hz": {
        "lng_range": [120.079320, 120.203163],
        "lat_range": [30.243830, 30.350527],
        "pickup_csv": "data/LADE/pickup/pickup_hz.csv",
        "delivery_csv": "data/LADE/delivery/delivery_hz.csv",
        "n_station": 6,
        # "station_id": [8, 10, 29, 20, 21, 33],
        "station_id": [38, 12, 10, 21, 33, 32] # 35
    },
    "cq": {
        "lng_range": [106.422522, 106.546171],
        "lat_range": [29.453564, 29.560914],
        "pickup_csv": "data/LADE/pickup/pickup_cq.csv",
        "delivery_csv": "data/LADE/delivery/delivery_cq.csv",
        "n_station": 6,
        # "station_id": [12, 1, 15, 29, 34, 20], # 效果好但是得改
        # "station_id": [12, 23, 34, 15, 6, 20],
        "station_id": [10, 23, 29, 34, 17, 21],
    },
    "sh": {
        "lng_range": [121.424605, 121.550168],
        "lat_range": [31.195720, 31.299977],
        "pickup_csv": "data/LADE/pickup/pickup_sh.csv",
        "delivery_csv": "data/LADE/delivery/delivery_sh.csv",
        "n_station": 6,
        "station_id": [28, 1, 16, 21, 2, 37],
        # "station_id": [28, 1, 16, 21, 2, 37],
        # "station_id": [8, 12, 35, 10, 21, 33, 32]
    }
}

def meters_to_degrees(grid_m, at_lat):
    """
    以给定纬度 at_lat（度）把米数转换为经度/纬度的度数近似值。
    近似：1 deg latitude ≈ 111320 m
           1 deg longitude ≈ 111320 * cos(lat_rad) m
    返回 (delta_lng_deg, delta_lat_deg)
    """
    lat_rad = math.radians(at_lat)
    deg_per_m_lat = 1.0 / 111320.0  # deg per meter for latitude
    deg_per_m_lng = 1.0 / (111320.0 * math.cos(lat_rad))
    delta_lat = grid_m * deg_per_m_lat
    delta_lng = grid_m * deg_per_m_lng
    return delta_lng, delta_lat

def build_grid(lng_min, lng_max, lat_min, lat_max, delta_lng, delta_lat):
    """
    构建网格索引与网格中心经纬，返回：
      - grid_lng_edges, grid_lat_edges （数组，包含边界）
      - ncols, nrows
    """
    # 保证包含右上边界：使用 np.arange 到超过 max 再裁切
    lng_edges = np.arange(lng_min, lng_max + delta_lng*0.5, delta_lng)
    lat_edges = np.arange(lat_min, lat_max + delta_lat*0.5, delta_lat)
    # 若最后一个边界小于 max，则追加
    if len(lng_edges) > 0 and lng_edges[-1] < lng_max:
        lng_edges = np.append(lng_edges, lng_max)
    if len(lat_edges) > 0 and lat_edges[-1] < lat_max:
        lat_edges = np.append(lat_edges, lat_max)
    ncols = len(lng_edges) - 1
    nrows = len(lat_edges) - 1
    return lng_edges, lat_edges, ncols, nrows

def assign_points_to_grid(df, lng_col, lat_col, lng_edges, lat_edges):
    """
    将点分配到网格。返回 counts 矩阵（shape nrows x ncols，row 0 = 最南边 -> 最北边）
    以及每点对应的 (row_idx, col_idx)（-1 表示落在网格外）
    """
    # 使用 np.searchsorted 得到所属区间 index
    lng_vals = df[lng_col].to_numpy()
    lat_vals = df[lat_col].to_numpy()
    # indices: col = searchsorted(lng_edges, x) - 1
    col_idx = np.searchsorted(lng_edges, lng_vals, side='right') - 1
    row_idx = np.searchsorted(lat_edges, lat_vals, side='right') - 1
    # 标记超出范围的点
    outside_mask = (col_idx < 0) | (col_idx >= (len(lng_edges)-1)) | (row_idx < 0) | (row_idx >= (len(lat_edges)-1))
    # 把超出的标为 -1
    col_idx[outside_mask] = -1
    row_idx[outside_mask] = -1
    # 统计
    nrows = len(lat_edges) - 1
    ncols = len(lng_edges) - 1
    counts = np.zeros((nrows, ncols), dtype=int)
    for r, c in zip(row_idx, col_idx):
        if r >= 0 and c >= 0:
            counts[r, c] += 1
    # 注意 searchsorted 返回的是基于 edges 从南->北、从西->东的索引
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
        
        # 环境实体设置
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
        
        # 加载数据集
        self.load_dataset()
        
        if self.is_train:
            self.batch_size = kwargs["sample_batch_size"]
        else:
            # 使用测试集数据
            test_from = self.cached.get("test_requests_from", self.cached.get("train_requests_from"))
            self.batch_size = min(test_from.shape[0], kwargs["sample_batch_size"])
        
        self.info_batch_size = self.batch_size if self.env_args["info_batch_size"] == -1 else self.env_args["info_batch_size"]
        
        self.batch_arange = torch.arange(self.batch_size, device=self.device)
        self.node_arange = torch.arange(self.n_node, device=self.device)
        self.requests_arange = torch.arange(self.n_tot_requests, device=self.device)
    
    def generate_filename(self):
        """根据参数生成数据集文件名"""
        split = "train" if self.is_train else "test"
        return f"{self.city}_{split}_node{self.n_node}_station{self.n_station}_totreq{self.n_tot_requests}_mindist{self.min_direct_dist}.pt"
    
    def load_dataset(self):
        """从预生成的数据集文件中加载数据"""
        # 构建文件路径
        base_dir = Path("/mnt/jfs6/g-bairui/results")
        processed_dir = base_dir / "data" / "LADE" / "processed"
        filename = self.generate_filename()
        file_path = processed_dir / filename
        
        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(
                f"数据集文件不存在: {file_path}\n"
                f"请先运行 data/LADE/lade_process.py 生成数据集。\n"
                f"期望的文件名格式: {{city}}_{{split}}_node{{n_node}}_station{{n_station}}_totreq{{n_tot_requests}}_mindist{{min_direct_dist}}.pt"
            )
        
        # 加载数据
        print(f"加载数据集: {file_path}")
        data = torch.load(file_path, map_location=self.device, weights_only=False)
        
        # 验证元数据
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
                    mismatches.append(f"{key}: 期望 {expected_value}, 实际 {actual_value}")
        
        if mismatches:
            raise ValueError(
                f"数据集元数据不匹配:\n" + "\n".join(mismatches) + 
                f"\n文件路径: {file_path}"
            )
        
        # 将数据加载到缓存
        self.cached = {}
        self.cached["coord"] = data["coord"].to(self.device)
        self.cached["node_node"] = data["node_node"].to(self.device)
        self.cached["station_idx"] = data["station_idx"].to(self.device)
        self.cached["station_mask"] = data["station_mask"].to(self.device)
        self.cached["station_node_node"] = data["station_node_node"].to(self.device)
        
        # 根据is_train选择训练集或测试集数据
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
        
        print(f"数据集加载完成！实例数: {metadata.get('n_instances', 'unknown')}")
    
    def reset(self, **kwargs):
        """重置环境，根据is_train选择训练/测试集数据"""
        if self.debug:
            self.rng.manual_seed(123321)
            print("use debug seed 123321.")
        
        # 选择数据集
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
        
        # 随机选择实例
        n_instances = requests_from.shape[0]
        if self.is_train:
            batch_idx = torch.randint(0, n_instances, (self.batch_size,), generator=self.rng, device=self.device)
        else:
            # 测试时使用固定种子保证可复现性
            rng_test = torch.Generator(self.device)
            rng_test.manual_seed(0)
            batch_idx = torch.randperm(n_instances, generator=rng_test, device=self.device)[:self.batch_size]
        
        # 设置节点坐标
        self.nodes = {
            "coord": self.cached["coord"].repeat(self.batch_size, 1, 1),
        }
        
        # 设置站点信息
        station_idx = self.cached["station_idx"].unsqueeze(0).repeat(self.batch_size, 1)
        self.station_idx = station_idx
        self.station_mask = self.cached["station_mask"].unsqueeze(0).repeat(self.batch_size, 1)
        self.station_node_node = self.cached["station_node_node"].unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        # 初始化载具
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
        
        # 初始化无人机
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
        
        # 设置订单
        visible = torch.full([self.batch_size, self.n_tot_requests], False, device=self.device)
        initial_counts = frame_accumulate[batch_idx, 0]  # 第一个时间步的订单数
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
        
        # 设置关系矩阵
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
        
        # 设置全局状态
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
        
        # 获取观察
        self.pre_obs, self.pre_unobs = self.get_obs()
        return self.pre_obs
    
    def seed(self, seed, seed_env):
        """设置随机种子"""
        self.rng.manual_seed(seed)
        self.rng_env.manual_seed(seed_env)

