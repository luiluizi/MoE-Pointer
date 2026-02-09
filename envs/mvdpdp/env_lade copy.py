import torch
import pandas as pd
import numpy as np
import math
import os
from pathlib import Path
import matplotlib.pyplot as plt

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
        "delivery_csv": "data/LADE/delivery/delivery_hz.csv"
    },
    "cq": {
        "lng_range": [106.452522, 106.576171],
        "lat_range": [29.483564, 29.590914],
        "pickup_csv": "data/LADE/pickup/pickup_cq.csv",
        "delivery_csv": "data/LADE/delivery/delivery_cq.csv"
    },
    "sh": {
        "lng_range": [121.424605, 121.550168],
        "lat_range": [31.195720, 31.299977],
        "pickup_csv": "data/LADE/pickup/pickup_sh.csv",
        "delivery_csv": "data/LADE/delivery/delivery_sh.csv"
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
        
        self.drone_speed_radio = 4
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
        
        # 网格大小 (1km)
        self.grid_size_m = 1000.0
        self.pickup_node_ratio = 0.4
        self.rng.manual_seed(kwargs["seed"])
        self.rng_env.manual_seed(kwargs["seed_env"])
        # 调用主生成流程
        self.genearte()
        
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
        
    
    def load_data(self):
        """加载pickup和delivery CSV数据"""
        config = CITY_CONFIGS[self.city]
        base_dir = Path(__file__).parent.parent.parent
        
        pickup_path = base_dir / config["pickup_csv"]
        delivery_path = base_dir / config["delivery_csv"]
        
        # 读取CSV文件
        pickup_df = pd.read_csv(pickup_path)
        delivery_df = pd.read_csv(delivery_path)
        
        # 查找经纬度字段
        lng_field_candidates = ["pickup_gps_lng", "lng", "longitude", "lon"]
        lat_field_candidates = ["pickup_gps_lat", "lat", "latitude"]
        
        pickup_lng_col = None
        pickup_lat_col = None
        for c in lng_field_candidates:
            if c in pickup_df.columns:
                pickup_lng_col = c
                break
        for c in lat_field_candidates:
            if c in pickup_df.columns:
                pickup_lat_col = c
                break
        
        if pickup_lng_col is None or pickup_lat_col is None:
            raise ValueError(f"Cannot find longitude/latitude columns in pickup CSV. Available: {pickup_df.columns.tolist()}")
        
        delivery_lng_field_candidates = ["delivery_gps_lng", "pickup_gps_lng", "lng", "longitude", "lon"]
        delivery_lat_field_candidates = ["delivery_gps_lat", "pickup_gps_lat", "lat", "latitude"]
        
        delivery_lng_col = None
        delivery_lat_col = None
        for c in delivery_lng_field_candidates:
            if c in delivery_df.columns:
                delivery_lng_col = c
                break
        for c in delivery_lat_field_candidates:
            if c in delivery_df.columns:
                delivery_lat_col = c
                break
        
        if delivery_lng_col is None or delivery_lat_col is None:
            raise ValueError(f"Cannot find longitude/latitude columns in delivery CSV. Available: {delivery_df.columns.tolist()}")
        
        # 提取经纬度数据
        pickup_df = pickup_df[[pickup_lng_col, pickup_lat_col]].copy()
        delivery_df = delivery_df[[delivery_lng_col, delivery_lat_col]].copy()
        
        # 处理缺失值
        pickup_df = pickup_df.dropna(subset=[pickup_lng_col, pickup_lat_col])
        delivery_df = delivery_df.dropna(subset=[delivery_lng_col, delivery_lat_col])
        
        # 转换为数值类型
        pickup_df[pickup_lng_col] = pd.to_numeric(pickup_df[pickup_lng_col], errors='coerce')
        pickup_df[pickup_lat_col] = pd.to_numeric(pickup_df[pickup_lat_col], errors='coerce')
        delivery_df[delivery_lng_col] = pd.to_numeric(delivery_df[delivery_lng_col], errors='coerce')
        delivery_df[delivery_lat_col] = pd.to_numeric(delivery_df[delivery_lat_col], errors='coerce')
        
        # 再次删除缺失值
        pickup_df = pickup_df.dropna(subset=[pickup_lng_col, pickup_lat_col])
        delivery_df = delivery_df.dropna(subset=[delivery_lng_col, delivery_lat_col])
        
        # 统一列名
        pickup_df = pickup_df.rename(columns={pickup_lng_col: 'lng', pickup_lat_col: 'lat'})
        delivery_df = delivery_df.rename(columns={delivery_lng_col: 'lng', delivery_lat_col: 'lat'})
        
        # 过滤数据范围
        lng_min, lng_max = config["lng_range"]
        lat_min, lat_max = config["lat_range"]
        
        pickup_df = pickup_df[
            (pickup_df['lng'] >= lng_min) & (pickup_df['lng'] <= lng_max) &
            (pickup_df['lat'] >= lat_min) & (pickup_df['lat'] <= lat_max)
        ].copy()
        
        delivery_df = delivery_df[
            (delivery_df['lng'] >= lng_min) & (delivery_df['lng'] <= lng_max) &
            (delivery_df['lat'] >= lat_min) & (delivery_df['lat'] <= lat_max)
        ].copy()
        
        return pickup_df, delivery_df, lng_min, lng_max, lat_min, lat_max
    
    def compute_daily_heatmaps(self, pickup_df, delivery_df, lng_min, lng_max, lat_min, lat_max):
        """计算每日热力图"""
        # 检查是否有日期字段
        # 如果没有日期字段，假设所有数据在同一天
        # 这里需要根据实际数据格式调整
        # 暂时假设数据中有日期字段，如果没有，则全部数据算作一天
        
        # 计算网格
        center_lat = (lat_min + lat_max) / 2.0
        delta_lng, delta_lat = meters_to_degrees(self.grid_size_m, center_lat)
        lng_edges, lat_edges, ncols, nrows = build_grid(lng_min, lng_max, lat_min, lat_max, delta_lng, delta_lat)
        
        # 计算网格中心坐标
        grid_centers = []
        for r in range(nrows):
            for c in range(ncols):
                lat_c = (lat_edges[r] + lat_edges[r+1]) / 2.0
                lng_c = (lng_edges[c] + lng_edges[c+1]) / 2.0
                grid_idx = r * ncols + c  # 一维索引
                grid_centers.append((lng_c, lat_c, r, c, grid_idx))
        
        # 检查是否有日期字段
        date_col = None
        date_candidates = ['date', 'order_date', 'day', 'order_day', 'time', 'order_time']
        
        # 对于pickup和delivery，分别处理日期
        # 如果没有日期字段，假设所有数据在同一天（日期为0）
        pickup_dates = None
        delivery_dates = None
        
        # 暂时将所有数据视为同一天的数据
        # 后续可以根据实际数据格式调整
        daily_heatmaps = {}
        
        # 计算整体热力图
        pickup_counts, _, _ = assign_points_to_grid(pickup_df, 'lng', 'lat', lng_edges, lat_edges)
        delivery_counts, _, _ = assign_points_to_grid(delivery_df, 'lng', 'lat', lng_edges, lat_edges)
        
        # 假设所有数据在同一天（日期索引为0）
        daily_heatmaps[0] = {
            'pickup': pickup_counts,
            'delivery': delivery_counts,
            'grid_centers': grid_centers,
            'lng_edges': lng_edges,
            'lat_edges': lat_edges,
            'nrows': nrows,
            'ncols': ncols
        }
        
        return daily_heatmaps
    
    def select_nodes(self, daily_heatmaps, selected_dates):
        """按40%/60%比例从累计热力图中选择pickup和delivery节点"""
        # 累计热力图
        cumulative_pickup = None
        cumulative_delivery = None
        
        for date in selected_dates:
            if date in daily_heatmaps:
                if cumulative_pickup is None:
                    cumulative_pickup = daily_heatmaps[date]['pickup'].copy()
                    cumulative_delivery = daily_heatmaps[date]['delivery'].copy()
                else:
                    cumulative_pickup += daily_heatmaps[date]['pickup']
                    cumulative_delivery += daily_heatmaps[date]['delivery']
        
        if cumulative_pickup is None:
            raise ValueError("No valid dates found in daily_heatmaps")
        
        # 计算节点数量
        n_pickup_nodes = int(self.n_node * self.pickup_node_ratio)
        n_delivery_nodes = self.n_node - n_pickup_nodes
        
        # 获取网格中心坐标
        grid_info = daily_heatmaps[selected_dates[0]]
        grid_centers = grid_info['grid_centers']
        nrows = grid_info['nrows']
        ncols = grid_info['ncols']
        
        # 选择pickup节点（按热度排序）
        pickup_heatmap_flat = cumulative_pickup.flatten()
        top_pickup_indices_flat = np.argsort(pickup_heatmap_flat)[-n_pickup_nodes:]
        
        # 选择delivery节点（排除已选pickup节点）
        delivery_heatmap_flat = cumulative_delivery.flatten()
        available_indices = [i for i in range(len(delivery_heatmap_flat)) 
                           if i not in top_pickup_indices_flat]
        if len(available_indices) < n_delivery_nodes:
            # 如果可用节点不足，从所有节点中选择
            top_delivery_indices_flat = np.argsort(delivery_heatmap_flat)[-n_delivery_nodes:]
        else:
            available_heatmap_vals = [delivery_heatmap_flat[i] for i in available_indices]
            top_available_indices = np.argsort(available_heatmap_vals)[-n_delivery_nodes:]
            top_delivery_indices_flat = [available_indices[i] for i in top_available_indices]
        
        # 转换为网格坐标
        pickup_nodes = []
        delivery_nodes = []
        
        for idx in top_pickup_indices_flat:
            if idx < len(grid_centers):
                pickup_nodes.append(grid_centers[idx])
        
        for idx in top_delivery_indices_flat:
            if idx < len(grid_centers):
                delivery_nodes.append(grid_centers[idx])
        
        # 合并节点
        all_nodes = pickup_nodes + delivery_nodes
        
        # 提取节点坐标（lng, lat）
        node_coords = np.array([(node[0], node[1]) for node in all_nodes])  # (n_node, 2)
        
        # 创建pickup和delivery节点mask
        pickup_mask = np.zeros(self.n_node, dtype=bool)
        delivery_mask = np.zeros(self.n_node, dtype=bool)
        pickup_mask[:n_pickup_nodes] = True
        delivery_mask[n_pickup_nodes:] = True
        
        # 创建节点索引映射（网格(r,c) -> 节点索引）
        grid_to_node = {}
        for node_idx, node_info in enumerate(all_nodes):
            if len(node_info) >= 4:
                r, c = node_info[2], node_info[3]
                grid_to_node[(r, c)] = node_idx
        
        return node_coords, grid_to_node, nrows, ncols, pickup_mask, delivery_mask
    
    def compute_distance_matrix(self, node_coords):
        """基于节点坐标计算曼哈顿距离矩阵"""
        # 参考env_for_DHRD.py的距离计算方式
        # 使用经纬度差值的绝对值之和作为曼哈顿距离
        # 转换为整数（以网格为单位，1km=1单位）
        n_node = len(node_coords)
        
        # 计算经纬度差值
        coords = node_coords  # (n_node, 2) where 2 is (lng, lat)
        lng_diff = np.abs(coords[:, 0:1] - coords[:, 0])  # (n_node, n_node)
        lat_diff = np.abs(coords[:, 1:2] - coords[:, 1])  # (n_node, n_node)
        
        # 曼哈顿距离
        manhattan_dist = lng_diff + lat_diff
        
        # 转换为1km单位的整数距离
        # 1度经度/纬度约等于111.32km，所以1km约等于1/111.32度
        # 对于1km网格，直接使用度数差乘以111.32，然后取整
        km_per_deg = 111.32
        node_node = (manhattan_dist * km_per_deg + 0.5).astype(int)
        
        # 对角线设为0
        np.fill_diagonal(node_node, 0)
        
        return node_node
    
    def generate_orders(self, daily_heatmaps, selected_dates, node_coords, grid_to_node, 
                       node_node, nrows, ncols, pickup_mask, delivery_mask, station_mask):
        """基于每日热力图采样订单
        订单起点：根据所有节点的pickup热力图选择（所有非站点节点都可以作为起点）
        订单终点：根据所有节点的delivery热力图选择（所有非站点节点都可以作为终点）
        """
        all_orders = []
        
        # 计算可用的节点（所有非站点节点都可以作为起点或终点）
        available_nodes = np.where(~station_mask)[0]
        
        if len(available_nodes) < 2:
            raise ValueError(f"Not enough available nodes (non-station): {len(available_nodes)}, need at least 2")
        
        for date in selected_dates:
            if date not in daily_heatmaps:
                continue
                
            heatmap_info = daily_heatmaps[date]
            pickup_heatmap = heatmap_info['pickup']
            delivery_heatmap = heatmap_info['delivery']
            grid_centers = heatmap_info['grid_centers']
            
            # 将热力图映射到所有节点的概率（包括pickup和delivery节点）
            pickup_node_probs = np.zeros(self.n_node)
            delivery_node_probs = np.zeros(self.n_node)
            
            # 计算每个节点对应的网格总热度
            for r in range(nrows):
                for c in range(ncols):
                    grid_idx = r * ncols + c
                    if (r, c) in grid_to_node:
                        node_idx = grid_to_node[(r, c)]
                        pickup_node_probs[node_idx] += pickup_heatmap[r, c]
                        delivery_node_probs[node_idx] += delivery_heatmap[r, c]
            
            # 对pickup热力图归一化（只考虑非站点节点）
            pickup_probs_available = pickup_node_probs[available_nodes]
            if pickup_probs_available.sum() > 0:
                pickup_node_probs[available_nodes] = pickup_probs_available / pickup_probs_available.sum()
            else:
                # 如果所有pickup热力为0，均匀分布
                pickup_node_probs[available_nodes] = 1.0 / len(available_nodes)
            pickup_node_probs[station_mask] = 0   # 站点概率为0
            
            # 对delivery热力图归一化（只考虑非站点节点）
            delivery_probs_available = delivery_node_probs[available_nodes]
            if delivery_probs_available.sum() > 0:
                delivery_node_probs[available_nodes] = delivery_probs_available / delivery_probs_available.sum()
            else:
                # 如果所有delivery热力为0，均匀分布
                delivery_node_probs[available_nodes] = 1.0 / len(available_nodes)
            delivery_node_probs[station_mask] = 0  # 站点概率为0
            
            # 生成订单
            orders_for_date = []
            
            # 预计算有效的节点对（起点和终点都是非站点，距离满足要求）
            valid_pairs = []
            for i in available_nodes:
                for j in available_nodes:
                    if i != j and node_node[i, j] >= self.min_direct_dist:
                        valid_pairs.append((i, j))
            
            if len(valid_pairs) == 0:
                print(f"Warning: No valid node pairs found for date {date}, using relaxed distance")
                # 降低距离要求
                for i in available_nodes:
                    for j in available_nodes:
                        if i != j and node_node[i, j] >= max(1, self.min_direct_dist - 3):
                            valid_pairs.append((i, j))
            
            # 如果仍然没有有效对，使用所有非站点节点对（忽略距离要求）
            if len(valid_pairs) == 0:
                for i in available_nodes:
                    for j in available_nodes:
                        if i != j:
                            valid_pairs.append((i, j))
            
            # 按概率采样节点对
            if len(valid_pairs) > 0:
                # 计算每个有效对的概率（起点根据pickup热力图，终点根据delivery热力图）
                pair_probs = np.array([
                    pickup_node_probs[pair[0]] * delivery_node_probs[pair[1]]
                    for pair in valid_pairs
                ])
                if pair_probs.sum() > 0:
                    pair_probs = pair_probs / pair_probs.sum()
                else:
                    pair_probs = np.ones(len(valid_pairs)) / len(valid_pairs)
                
                # 采样订单 - 使用torch的multinomial替代np.random.choice以保证可复现性
                n_samples = min(self.n_tot_requests, len(valid_pairs))
                if n_samples > 0:
                    # 将概率转换为torch tensor
                    pair_probs_tensor = torch.from_numpy(pair_probs).to(self.device).float()
                    # 使用torch.multinomial进行采样
                    sampled_indices_tensor = torch.multinomial(
                        pair_probs_tensor, 
                        n_samples, 
                        replacement=True, 
                        generator=self.rng
                    )
                    sampled_indices = sampled_indices_tensor.cpu().numpy()
                    for idx in sampled_indices:
                        from_node, to_node = valid_pairs[idx]
                        # 验证：确保起点和终点都不是站点
                        assert not station_mask[from_node], \
                            f"Invalid pickup node: {from_node} is a station"
                        assert not station_mask[to_node], \
                            f"Invalid delivery node: {to_node} is a station"
                        orders_for_date.append((int(from_node), int(to_node), 1))
            
            if len(orders_for_date) < self.n_tot_requests:
                print(f"Warning: Only generated {len(orders_for_date)} orders for date {date}, expected {self.n_tot_requests}")
            
            all_orders.append(orders_for_date[:self.n_tot_requests])
        
        return all_orders
    
    def assign_order_times(self, n_orders):
        """分配订单出现时间"""
        # n_init_requests个在时间步0
        # 剩余n_norm_requests个在[1, n_requested_frame]内随机分配
        n_init = self.env_args["n_init_requests"]
        n_requested_frame = self.env_args["n_requested_frame"]
        
        appear_times = []
        if n_orders <= n_init:
            # 如果订单数少于初始订单数，全部在时间步0
            appear_times = [0] * n_orders
        else:
            # 前n_init个在时间步0
            appear_times = [0] * n_init
            # 剩余的在[1, n_requested_frame]内随机分配
            remaining = n_orders - n_init
            if n_requested_frame > 1:
                # 使用torch的随机数生成器以保证可复现性
                remaining_times_tensor = torch.randint(
                    1, n_requested_frame + 1, 
                    (remaining,), 
                    generator=self.rng, 
                    device=self.device
                )
                remaining_times = remaining_times_tensor.cpu().numpy().tolist()
            else:
                remaining_times = [1] * remaining if remaining > 0 else []
            appear_times.extend(remaining_times)
        
        return np.array(appear_times, dtype=np.int64)
    
    def select_stations(self, node_coords):
        """从节点中随机选择n_station个作为站点
        使用self.rng_env确保训练集和测试集的站点选择一致
        """
        n_node = len(node_coords)
        if self.n_station > n_node:
            raise ValueError(f"n_station ({self.n_station}) > n_node ({n_node})")
        
        # 使用torch的随机数生成器，确保训练集和测试集使用相同的随机种子时选择一致
        # 生成0到n_node-1的随机排列，然后取前n_station个
        perm = torch.randperm(n_node, generator=self.rng_env, device=self.device)
        station_indices = perm[:self.n_station].cpu().numpy()
        station_indices = np.sort(station_indices)
        
        # 添加哨兵点
        station_indices = np.concatenate([station_indices, [n_node]])
        
        return station_indices
    
    def plot_nodes_and_stations(self, node_coords, station_mask, city):
        """绘制节点和起降站分布图"""
        # 确保images文件夹存在
        base_dir = Path(__file__).parent.parent.parent
        images_dir = base_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # 提取坐标
        lng_coords = node_coords[:, 0]
        lat_coords = node_coords[:, 1]
        
        # 分离节点和起降站
        node_mask = ~station_mask
        nodes_lng = lng_coords[node_mask]
        nodes_lat = lat_coords[node_mask]
        stations_lng = lng_coords[station_mask]
        stations_lat = lat_coords[station_mask]
        
        # 创建图形
        plt.figure(figsize=(10, 10))
        
        # 绘制普通节点
        plt.scatter(nodes_lng, nodes_lat, c='blue', label='节点', alpha=0.6, s=30)
        
        # 绘制起降站
        plt.scatter(stations_lng, stations_lat, c='red', label='起降站', s=100, marker='s', edgecolors='black', linewidths=1.5)
        
        # 设置标签和标题
        plt.xlabel('经度', fontsize=12)
        plt.ylabel('纬度', fontsize=12)
        plt.title(f'{city.upper()} - 节点和起降站分布图', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        # 保存图片
        output_path = images_dir / f'nodes_stations_{city}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  节点和起降站分布图已保存至: {output_path}")
    
    def genearte(self):
        """主生成流程：整合所有步骤，生成训练/测试集订单并缓存"""
        print(f"开始生成LADE数据集环境 (city={self.city})...")
        
        # 1. 加载数据
        print("步骤1: 加载数据...")
        pickup_df, delivery_df, lng_min, lng_max, lat_min, lat_max = self.load_data()
        print(f"  Pickup数据: {len(pickup_df)} 条记录")
        print(f"  Delivery数据: {len(delivery_df)} 条记录")
        
        # 2. 计算每日热力图
        print("步骤2: 计算每日热力图...")
        daily_heatmaps = self.compute_daily_heatmaps(pickup_df, delivery_df, lng_min, lng_max, lat_min, lat_max)
        print(f"  计算了 {len(daily_heatmaps)} 天的热力图")
        
        # 3. 确定日期范围（简化处理：假设所有数据在同一天，日期索引为0）
        # 实际应该根据数据中的日期字段分组，这里先简化
        all_dates = sorted(daily_heatmaps.keys())
        n_total_days = len(all_dates)
        
        # 如果没有日期信息，假设只有一天
        if n_total_days == 0:
            raise ValueError("No daily heatmaps computed")
        
        # 划分训练集和测试集日期
        # 如果只有一天，则训练和测试都用这一天
        if n_total_days == 1:
            train_dates = all_dates
            test_dates = all_dates
        else:
            train_dates = all_dates[:min(self.train_days, n_total_days)]
            test_start = min(self.train_days, n_total_days)
            test_end = min(test_start + self.test_days, n_total_days)
            test_dates = all_dates[test_start:test_end]
        
        print(f"  训练集日期: {len(train_dates)} 天")
        print(f"  测试集日期: {len(test_dates)} 天")
        
        # 4. 选择节点（使用训练集日期）
        print("步骤3: 选择节点...")
        node_coords, grid_to_node, nrows, ncols, pickup_mask, delivery_mask = self.select_nodes(daily_heatmaps, train_dates)
        print(f"  选择了 {len(node_coords)} 个节点")
        print(f"  Pickup节点数: {pickup_mask.sum()}, Delivery节点数: {delivery_mask.sum()}")
        
        # 5. 计算距离矩阵
        print("步骤4: 计算距离矩阵...")
        node_node = self.compute_distance_matrix(node_coords)
        print(f"  距离矩阵范围: [{node_node.min()}, {node_node.max()}]")
        # 输出节点间平均距离（与env.py保持一致）
        node_node_tensor = torch.from_numpy(node_node).float()
        print(f"  node average dist {torch.mean(node_node_tensor).item():.2f}")
        
        # 6. 选择站点（从所有节点中随机选择）
        print("步骤5: 选择站点...")
        station_indices = self.select_stations(node_coords)
        station_mask = np.zeros(self.n_node, dtype=bool)
        station_mask[station_indices[:-1]] = True
        station_node_node = node_node[np.ix_(station_indices[:-1], station_indices[:-1])]
        print(f"  选择了 {len(station_indices)-1} 个站点")
        print(f"  站点中Pickup节点数: {(pickup_mask & station_mask).sum()}")
        print(f"  站点中Delivery节点数: {(delivery_mask & station_mask).sum()}")
        
        # 绘制节点和起降站分布图
        # self.plot_nodes_and_stations(node_coords, station_mask, self.city)
        
        # 7. 生成训练集订单
        print("步骤6: 生成训练集订单...")
        train_orders_list = self.generate_orders(
            daily_heatmaps, train_dates, node_coords, grid_to_node, node_node, nrows, ncols,
            pickup_mask, delivery_mask, station_mask
        )
        print(f"  生成了 {len(train_orders_list)} 天的训练订单")
        
        # 8. 生成测试集订单
        print("步骤7: 生成测试集订单...")
        test_orders_list = self.generate_orders(
            daily_heatmaps, test_dates, node_coords, grid_to_node, node_node, nrows, ncols,
            pickup_mask, delivery_mask, station_mask
        )
        print(f"  生成了 {len(test_orders_list)} 天的测试订单")
        
        # 9. 处理订单数据并分配时间
        print("步骤8: 处理订单数据并分配时间...")
        n_frame = self.env_args["n_frame"]
        
        # 处理训练集订单
        train_instances = []
        train_frame_accumulate = []
        for day_idx, orders in enumerate(train_orders_list):
            if len(orders) == 0:
                continue
            
            # 分配出现时间
            appear_times = self.assign_order_times(len(orders))
            
            # 构建订单数组
            orders_array = np.array(orders)  # (n_orders, 3) where 3 is (from, to, value)
            requests_from = orders_array[:, 0].astype(np.int64)
            requests_to = orders_array[:, 1].astype(np.int64)
            
            # 排序订单按出现时间
            sort_idx = np.argsort(appear_times)
            requests_from = requests_from[sort_idx]
            requests_to = requests_to[sort_idx]
            appear_times = appear_times[sort_idx]
            
            # 扩展到n_tot_requests长度（padding）
            n_actual = len(requests_from)
            if n_actual < self.n_tot_requests:
                padding_len = self.n_tot_requests - n_actual
                requests_from = np.pad(requests_from, (0, padding_len), constant_values=0)
                requests_to = np.pad(requests_to, (0, padding_len), constant_values=0)
                appear_times = np.pad(appear_times, (0, padding_len), constant_values=n_frame)
            
            # 截断到n_tot_requests
            requests_from = requests_from[:self.n_tot_requests]
            requests_to = requests_to[:self.n_tot_requests]
            appear_times = appear_times[:self.n_tot_requests]
            
            train_instances.append({
                'from': requests_from,
                'to': requests_to,
                'appear': appear_times
            })
            
            # 计算累积请求数
            frame_counts = np.bincount(appear_times, minlength=n_frame + 1)
            frame_accumulate = np.cumsum(frame_counts)
            train_frame_accumulate.append(frame_accumulate)
        
        # 处理测试集订单
        test_instances = []
        test_frame_accumulate = []
        for day_idx, orders in enumerate(test_orders_list):
            if len(orders) == 0:
                continue
            
            # 分配出现时间
            appear_times = self.assign_order_times(len(orders))
            
            # 构建订单数组
            orders_array = np.array(orders)
            requests_from = orders_array[:, 0].astype(np.int64)
            requests_to = orders_array[:, 1].astype(np.int64)
            
            # 排序订单按出现时间
            sort_idx = np.argsort(appear_times)
            requests_from = requests_from[sort_idx]
            requests_to = requests_to[sort_idx]
            appear_times = appear_times[sort_idx]
            
            # 扩展到n_tot_requests长度（padding）
            n_actual = len(requests_from)
            if n_actual < self.n_tot_requests:
                padding_len = self.n_tot_requests - n_actual
                requests_from = np.pad(requests_from, (0, padding_len), constant_values=0)
                requests_to = np.pad(requests_to, (0, padding_len), constant_values=0)
                appear_times = np.pad(appear_times, (0, padding_len), constant_values=n_frame)
            
            # 截断到n_tot_requests
            requests_from = requests_from[:self.n_tot_requests]
            requests_to = requests_to[:self.n_tot_requests]
            appear_times = appear_times[:self.n_tot_requests]
            
            test_instances.append({
                'from': requests_from,
                'to': requests_to,
                'appear': appear_times
            })
            
            # 计算累积请求数
            frame_counts = np.bincount(appear_times, minlength=n_frame + 1)
            frame_accumulate = np.cumsum(frame_counts)
            test_frame_accumulate.append(frame_accumulate)
        
        # 确保至少有一个实例
        if len(train_instances) == 0:
            raise ValueError("No train instances generated")
        if len(test_instances) == 0:
            test_instances = train_instances
            test_frame_accumulate = train_frame_accumulate
        
        # 10. 转换为tensor并缓存
        print("步骤9: 缓存数据...")
        n_train_inst = len(train_instances)
        n_test_inst = len(test_instances)
        
        # 构建训练集tensor
        train_requests_from = torch.zeros(n_train_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        train_requests_to = torch.zeros(n_train_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        train_requests_value = torch.zeros(n_train_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        train_frame_accumulate_tensor = torch.zeros(n_train_inst, n_frame + 1, dtype=torch.int64, device=self.device)
        
        for i, inst in enumerate(train_instances):
            train_requests_from[i] = torch.from_numpy(inst['from']).to(self.device)
            train_requests_to[i] = torch.from_numpy(inst['to']).to(self.device)
            train_frame_accumulate_tensor[i] = torch.from_numpy(train_frame_accumulate[i]).to(self.device)
        
        # 构建测试集tensor
        test_requests_from = torch.zeros(n_test_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        test_requests_to = torch.zeros(n_test_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        test_requests_value = torch.zeros(n_test_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        test_frame_accumulate_tensor = torch.zeros(n_test_inst, n_frame + 1, dtype=torch.int64, device=self.device)
        
        for i, inst in enumerate(test_instances):
            test_requests_from[i] = torch.from_numpy(inst['from']).to(self.device)
            test_requests_to[i] = torch.from_numpy(inst['to']).to(self.device)
            test_frame_accumulate_tensor[i] = torch.from_numpy(test_frame_accumulate[i]).to(self.device)
        
        # 计算订单value（等于起点和终点之间的曼哈顿距离）
        node_node_tensor = torch.from_numpy(node_node).to(self.device).long()
        
        # 训练集value计算：订单价值 = 曼哈顿距离
        train_from_valid = (train_requests_from < self.n_node) & (train_requests_to < self.n_node)
        train_distances = node_node_tensor[train_requests_from.clamp(0, self.n_node-1), 
                                          train_requests_to.clamp(0, self.n_node-1)]
        train_requests_value = train_distances.long()  # 直接使用曼哈顿距离作为订单价值
        train_requests_value = train_requests_value * train_from_valid.long()  # 无效订单的value设为0
        
        # 测试集value计算：订单价值 = 曼哈顿距离
        test_from_valid = (test_requests_from < self.n_node) & (test_requests_to < self.n_node)
        test_distances = node_node_tensor[test_requests_from.clamp(0, self.n_node-1), 
                                         test_requests_to.clamp(0, self.n_node-1)]
        test_requests_value = test_distances.long()  # 直接使用曼哈顿距离作为订单价值
        test_requests_value = test_requests_value * test_from_valid.long()  # 无效订单的value设为0
        
        # 缓存所有数据
        self.cached = {}
        self.cached["coord"] = torch.from_numpy(node_coords).to(self.device)  # (n_node, 2)
        self.cached["node_node"] = torch.from_numpy(node_node).to(self.device)  # (n_node, n_node)
        self.cached["station_idx"] = torch.from_numpy(station_indices).to(self.device)  # (n_station+1,)
        self.cached["station_mask"] = torch.from_numpy(station_mask).to(self.device)  # (n_node,)
        self.cached["station_node_node"] = torch.from_numpy(station_node_node).to(self.device)  # (n_station, n_station)
        
        # 训练集数据
        self.cached["train_requests_from"] = train_requests_from
        self.cached["train_requests_to"] = train_requests_to
        self.cached["train_requests_value"] = train_requests_value
        self.cached["train_frame_accumulate"] = train_frame_accumulate_tensor
        
        # 测试集数据
        self.cached["test_requests_from"] = test_requests_from
        self.cached["test_requests_to"] = test_requests_to
        self.cached["test_requests_value"] = test_requests_value
        self.cached["test_frame_accumulate"] = test_frame_accumulate_tensor
        
        print("数据生成完成！")
        print(f"  训练实例数: {n_train_inst}")
        print(f"  测试实例数: {n_test_inst}")
    
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

