#!/usr/bin/env python3
"""
LADE数据集预处理脚本
从CSV数据生成训练集和测试集，并保存为.pt文件
"""
import torch
import pandas as pd
import numpy as np
import math
import argparse
import yaml
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入辅助函数
from envs.env_lade import (
    CITY_CONFIGS, meters_to_degrees, build_grid, assign_points_to_grid
)

# 固定参数
GRID_SIZE_M = 1000.0
PICKUP_NODE_RATIO = 0.4
DEFAULT_SEED = 0
DEFAULT_SEED_ENV = 1


class LADEDataGenerator:
    """LADE数据集生成器"""
    
    def __init__(self, city, n_node, n_station, n_tot_requests, min_direct_dist,
                 n_frame, n_requested_frame, n_init_requests, n_norm_requests,
                 train_days=60, test_days=20, seed=DEFAULT_SEED, seed_env=DEFAULT_SEED_ENV,
                 device='cpu'):
        self.city = city
        self.n_node = n_node
        self.n_station = n_station
        self.n_tot_requests = n_tot_requests
        self.min_direct_dist = min_direct_dist
        self.n_frame = n_frame
        self.n_requested_frame = n_requested_frame
        self.n_init_requests = n_init_requests
        self.n_norm_requests = n_norm_requests
        self.train_days = train_days
        self.test_days = test_days
        self.grid_size_m = GRID_SIZE_M
        self.pickup_node_ratio = PICKUP_NODE_RATIO
        
        self.device = torch.device(device)
        self.rng = torch.Generator(self.device)
        self.rng_env = torch.Generator(self.device)
        self.seed = seed
        self.seed_env = seed_env
        self.rng.manual_seed(seed)
        self.rng_env.manual_seed(seed_env)
    
    def load_data(self):
        """加载pickup和delivery CSV数据"""
        config = CITY_CONFIGS[self.city]
        base_dir = Path("/mnt/jfs6/g-bairui/results")
        
        pickup_path = base_dir / config["pickup_csv"]
        delivery_path = base_dir / config["delivery_csv"]
        
        # 读取CSV文件
        pickup_df = pd.read_csv(pickup_path)
        delivery_df = pd.read_csv(delivery_path)
        
        # 检查必需字段
        required_pickup_cols = ['pickup_gps_lng', 'pickup_gps_lat']
        required_delivery_cols = ['delivery_gps_lng', 'delivery_gps_lat']
        
        missing_pickup = [c for c in required_pickup_cols if c not in pickup_df.columns]
        missing_delivery = [c for c in required_delivery_cols if c not in delivery_df.columns]
        
        if missing_pickup:
            raise ValueError(f"Pickup CSV缺少必需字段: {missing_pickup}. 可用字段: {pickup_df.columns.tolist()}")
        if missing_delivery:
            raise ValueError(f"Delivery CSV缺少必需字段: {missing_delivery}. 可用字段: {delivery_df.columns.tolist()}")
        
        # 提取需要的列（包含ds字段如果存在）
        pickup_cols = required_pickup_cols + (['ds'] if 'ds' in pickup_df.columns else [])
        delivery_cols = required_delivery_cols + (['ds'] if 'ds' in delivery_df.columns else [])
        
        pickup_df = pickup_df[pickup_cols].copy()
        delivery_df = delivery_df[delivery_cols].copy()
        
        # 转换为数值类型并处理缺失值
        pickup_df['pickup_gps_lng'] = pd.to_numeric(pickup_df['pickup_gps_lng'], errors='coerce')
        pickup_df['pickup_gps_lat'] = pd.to_numeric(pickup_df['pickup_gps_lat'], errors='coerce')
        delivery_df['delivery_gps_lng'] = pd.to_numeric(delivery_df['delivery_gps_lng'], errors='coerce')
        delivery_df['delivery_gps_lat'] = pd.to_numeric(delivery_df['delivery_gps_lat'], errors='coerce')
        
        # 删除缺失值
        pickup_df = pickup_df.dropna(subset=['pickup_gps_lng', 'pickup_gps_lat'])
        delivery_df = delivery_df.dropna(subset=['delivery_gps_lng', 'delivery_gps_lat'])
        
        # 统一列名
        pickup_df = pickup_df.rename(columns={'pickup_gps_lng': 'lng', 'pickup_gps_lat': 'lat'})
        delivery_df = delivery_df.rename(columns={'delivery_gps_lng': 'lng', 'delivery_gps_lat': 'lat'})
        
        # 处理日期字段（ds字段，格式如0618, 0814, 0901）
        if 'ds' in pickup_df.columns:
            pickup_df['ds'] = pickup_df['ds'].astype(str)
            pickup_df = pickup_df[pickup_df['ds'].notna() & (pickup_df['ds'] != '')]
        else:
            pickup_df['ds'] = None
            
        if 'ds' in delivery_df.columns:
            delivery_df['ds'] = delivery_df['ds'].astype(str)
            delivery_df = delivery_df[delivery_df['ds'].notna() & (delivery_df['ds'] != '')]
        else:
            delivery_df['ds'] = None
        
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
        """计算每日热力图，按日期（ds字段）分组"""
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
                grid_idx = r * ncols + c
                grid_centers.append((lng_c, lat_c, r, c, grid_idx))
        
        daily_heatmaps = {}
        
        # 检查是否有日期信息
        pickup_has_date = 'ds' in pickup_df.columns and pickup_df['ds'].notna().any()
        delivery_has_date = 'ds' in delivery_df.columns and delivery_df['ds'].notna().any()
        
        if pickup_has_date or delivery_has_date:
            # 获取所有唯一日期并排序
            all_dates = set()
            if pickup_has_date:
                all_dates.update(pickup_df['ds'].dropna().unique())
            if delivery_has_date:
                all_dates.update(delivery_df['ds'].dropna().unique())
            all_dates = sorted(list(all_dates))
            
            # 按日期分组计算热力图
            for date_idx, date_val in enumerate(all_dates):
                pickup_day = pickup_df[pickup_df['ds'] == date_val] if pickup_has_date else pickup_df
                delivery_day = delivery_df[delivery_df['ds'] == date_val] if delivery_has_date else delivery_df
                
                pickup_counts, _, _ = assign_points_to_grid(pickup_day, 'lng', 'lat', lng_edges, lat_edges)
                delivery_counts, _, _ = assign_points_to_grid(delivery_day, 'lng', 'lat', lng_edges, lat_edges)
                
                daily_heatmaps[date_idx] = {
                    'pickup': pickup_counts,
                    'delivery': delivery_counts,
                    'grid_centers': grid_centers,
                    'lng_edges': lng_edges,
                    'lat_edges': lat_edges,
                    'nrows': nrows,
                    'ncols': ncols,
                    'ds': date_val
                }
        else:
            # 如果没有日期信息，将所有数据视为同一天
            pickup_counts, _, _ = assign_points_to_grid(pickup_df, 'lng', 'lat', lng_edges, lat_edges)
            delivery_counts, _, _ = assign_points_to_grid(delivery_df, 'lng', 'lat', lng_edges, lat_edges)
            
            daily_heatmaps[0] = {
                'pickup': pickup_counts,
                'delivery': delivery_counts,
                'grid_centers': grid_centers,
                'lng_edges': lng_edges,
                'lat_edges': lat_edges,
                'nrows': nrows,
                'ncols': ncols,
                'ds': None
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
            top_delivery_indices_flat = np.argsort(delivery_heatmap_flat)[-n_delivery_nodes:]
        else:
            available_heatmap_vals = [delivery_heatmap_flat[i] for i in available_indices]
            top_available_indices = np.argsort(available_heatmap_vals)[-n_delivery_nodes:]
            top_delivery_indices_flat = [available_indices[i] for i in top_available_indices]
        
        # 转换为网格坐标
        pickup_nodes = [grid_centers[idx] for idx in top_pickup_indices_flat if idx < len(grid_centers)]
        delivery_nodes = [grid_centers[idx] for idx in top_delivery_indices_flat if idx < len(grid_centers)]
        
        # 合并节点并提取坐标
        all_nodes = pickup_nodes + delivery_nodes
        node_coords = np.array([(node[0], node[1]) for node in all_nodes])
        
        # 创建节点索引映射（网格(r,c) -> 节点索引）
        grid_to_node = {(node[2], node[3]): idx for idx, node in enumerate(all_nodes) if len(node) >= 4}
        
        return node_coords, grid_to_node, nrows, ncols
    
    def compute_distance_matrix(self, node_coords):
        """基于节点坐标计算曼哈顿距离矩阵"""
        # 计算曼哈顿距离（经纬度差值）
        lng_diff = np.abs(node_coords[:, 0:1] - node_coords[:, 0])
        lat_diff = np.abs(node_coords[:, 1:2] - node_coords[:, 1])
        manhattan_dist = lng_diff + lat_diff
        
        # 转换为1km单位的整数距离
        node_node = (manhattan_dist * 111.32 + 0.5).astype(int)
        np.fill_diagonal(node_node, 0)
        
        return node_node
    
    def select_stations(self, node_coords, manual_station_indices=None):
        """从节点中选择n_station个作为站点
        
        Args:
            node_coords: 节点坐标数组
            manual_station_indices: 手动指定的站点索引列表，如果提供则使用，否则随机选择
        """
        n_node = len(node_coords)
        if self.n_station > n_node:
            raise ValueError(f"n_station ({self.n_station}) > n_node ({n_node})")
        
        # 如果提供了手动指定的站点索引，使用手动指定的
        if manual_station_indices is not None:
            station_indices = np.array(manual_station_indices, dtype=np.int64)
            
            # 验证索引有效性
            if len(station_indices) != self.n_station:
                raise ValueError(
                    f"手动指定的站点数量 ({len(station_indices)}) 与 n_station ({self.n_station}) 不匹配"
                )
            if np.any(station_indices < 0) or np.any(station_indices >= n_node):
                raise ValueError(
                    f"站点索引必须在 [0, {n_node-1}] 范围内，但发现: {station_indices}"
                )
            if len(np.unique(station_indices)) != len(station_indices):
                raise ValueError(f"站点索引不能重复: {station_indices}")
            
            station_indices = np.sort(station_indices)
            print(f"  使用手动指定的站点索引: {station_indices.tolist()}")
        else:
            # 否则使用随机选择
            perm = torch.randperm(n_node, generator=self.rng_env, device=self.device)
            station_indices = perm[:self.n_station].cpu().numpy()
            station_indices = np.sort(station_indices)
            print(f"  随机选择的站点索引: {station_indices.tolist()}")
        
        # 添加哨兵点
        station_indices = np.concatenate([station_indices, [n_node]])
        
        return station_indices
    
    def generate_orders(self, daily_heatmaps, selected_dates, node_coords, grid_to_node, 
                       node_node, nrows, ncols, station_mask):
        """基于每日热力图采样订单"""
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
            
            # 将热力图映射到所有节点的概率
            pickup_node_probs = np.zeros(self.n_node)
            delivery_node_probs = np.zeros(self.n_node)
            
            # 计算每个节点对应的网格总热度
            for r in range(nrows):
                for c in range(ncols):
                    if (r, c) in grid_to_node:
                        node_idx = grid_to_node[(r, c)]
                        pickup_node_probs[node_idx] += pickup_heatmap[r, c]
                        delivery_node_probs[node_idx] += delivery_heatmap[r, c]
            
            # 归一化概率（只考虑非站点节点）
            for probs in [pickup_node_probs, delivery_node_probs]:
                probs_available = probs[available_nodes]
                if probs_available.sum() > 0:
                    probs[available_nodes] = probs_available / probs_available.sum()
                else:
                    probs[available_nodes] = 1.0 / len(available_nodes)
                probs[station_mask] = 0
            
            # 生成订单
            orders_for_date = []
            
            # 预计算有效的节点对（逐步放宽距离限制）
            valid_pairs = [(i, j) for i in available_nodes for j in available_nodes 
                          if i != j and node_node[i, j] >= self.min_direct_dist]
            
            if len(valid_pairs) == 0:
                print(f"Warning: No valid node pairs found for date {date}, using relaxed distance")
                min_dist = max(1, self.min_direct_dist - 3)
                valid_pairs = [(i, j) for i in available_nodes for j in available_nodes 
                              if i != j and node_node[i, j] >= min_dist]
            
            if len(valid_pairs) == 0:
                valid_pairs = [(i, j) for i in available_nodes for j in available_nodes if i != j]
            
            # 按概率采样节点对
            if len(valid_pairs) > 0:
                pair_probs = np.array([
                    pickup_node_probs[pair[0]] * delivery_node_probs[pair[1]]
                    for pair in valid_pairs
                ])
                if pair_probs.sum() > 0:
                    pair_probs = pair_probs / pair_probs.sum()
                else:
                    pair_probs = np.ones(len(valid_pairs)) / len(valid_pairs)
                
                # 采样订单
                # n_samples = min(self.n_tot_requests, len(valid_pairs))
                n_samples = self.n_tot_requests
                if n_samples > 0:
                    pair_probs_tensor = torch.from_numpy(pair_probs).to(self.device).float()
                    sampled_indices_tensor = torch.multinomial(
                        pair_probs_tensor, 
                        n_samples, 
                        replacement=True, 
                        generator=self.rng
                    )
                    sampled_indices = sampled_indices_tensor.cpu().numpy()
                    for idx in sampled_indices:
                        from_node, to_node = valid_pairs[idx]
                        assert not station_mask[from_node], f"Invalid pickup node: {from_node} is a station"
                        assert not station_mask[to_node], f"Invalid delivery node: {to_node} is a station"
                        orders_for_date.append((int(from_node), int(to_node), 1))
            
            if len(orders_for_date) < self.n_tot_requests:
                print(f"Warning: Only generated {len(orders_for_date)} orders for date {date}, expected {self.n_tot_requests}")
            
            all_orders.append(orders_for_date[:self.n_tot_requests])
        
        return all_orders
    
    def assign_order_times(self, n_orders):
        """分配订单出现时间"""
        n_init = min(n_orders, self.n_init_requests)
        appear_times = [0] * n_init
        
        remaining = n_orders - n_init
        if remaining > 0:
            if self.n_requested_frame > 1:
                remaining_times = torch.randint(
                    1, self.n_requested_frame + 1, 
                    (remaining,), 
                    generator=self.rng, 
                    device=self.device
                ).cpu().numpy().tolist()
            else:
                remaining_times = [1] * remaining
            appear_times.extend(remaining_times)
        
        return np.array(appear_times, dtype=np.int64)
    
    def generate_dataset(self, split='train'):
        """生成数据集（训练集或测试集）"""
        print(f"开始生成LADE数据集 (city={self.city}, split={split})...")
        self.rng.manual_seed(self.seed)
        self.rng_env.manual_seed(self.seed_env)
        
        # 1. 加载数据
        print("步骤1: 加载数据...")
        pickup_df, delivery_df, lng_min, lng_max, lat_min, lat_max = self.load_data()
        print(f"  Pickup数据: {len(pickup_df)} 条记录")
        print(f"  Delivery数据: {len(delivery_df)} 条记录")
        
        # 2. 计算每日热力图
        print("步骤2: 计算每日热力图...")
        daily_heatmaps = self.compute_daily_heatmaps(pickup_df, delivery_df, lng_min, lng_max, lat_min, lat_max)
        print(f"  计算了 {len(daily_heatmaps)} 天的热力图")
        
        # 3. 确定日期范围
        all_dates = sorted(daily_heatmaps.keys())
        n_total_days = len(all_dates)
        
        if n_total_days == 0:
            raise ValueError("No daily heatmaps computed")
        
        # 划分训练集和测试集日期
        if n_total_days == 1:
            selected_dates = all_dates
        else:
            if split == 'train':
                selected_dates = all_dates[:min(self.train_days, n_total_days)]
            else:
                train_start = min(self.train_days, n_total_days)
                test_end = min(train_start + self.test_days, n_total_days)
                selected_dates = all_dates[train_start:test_end]
        
        print(f"  使用日期: {len(selected_dates)} 天")
        
        # 4. 选择节点（使用所有日期的累计热力图来选择节点，确保训练和测试使用相同的节点）
        print("步骤3: 选择节点（使用所有日期的累计热力图）...")
        node_coords, grid_to_node, nrows, ncols = self.select_nodes(daily_heatmaps, all_dates)
        print(f"  选择了 {len(node_coords)} 个节点")
        
        # 5. 计算距离矩阵
        print("步骤4: 计算距离矩阵...")
        node_node = self.compute_distance_matrix(node_coords)
        print(f"  距离矩阵范围: [{node_node.min()}, {node_node.max()}]")
        
        # 6. 选择站点（重置rng_env确保训练集和测试集使用相同的随机状态）
        print("步骤5: 选择站点...")
        # 从CITY_CONFIGS中读取station_id（如果存在）
        config = CITY_CONFIGS[self.city]
        manual_station_indices = None
        if 'station_id' in config and config['station_id'] is not None:
            manual_station_indices = config['station_id']
            # 验证station_id数量是否与n_station匹配
            if len(manual_station_indices) != self.n_station:
                raise ValueError(
                    f"CITY_CONFIGS中的station_id数量 ({len(manual_station_indices)}) 与 n_station ({self.n_station}) 不匹配"
                )
            print(f"  从CITY_CONFIGS读取到手动指定的站点索引: {manual_station_indices}")
        else:
            # 重置rng_env种子，确保训练集和测试集选择相同的station
            # 由于训练集和测试集使用相同的节点坐标，重置种子后选择结果会一致
            self.rng_env.manual_seed(self.seed_env)
            print(f"seed_env: {self.seed_env}")
        
        station_indices = self.select_stations(node_coords, manual_station_indices=manual_station_indices)
        station_mask = np.zeros(self.n_node, dtype=bool)
        station_mask[station_indices[:-1]] = True
        station_node_node = node_node[np.ix_(station_indices[:-1], station_indices[:-1])]
        print(f"  选择了 {len(station_indices)-1} 个站点")
        
        # 7. 生成订单
        print(f"步骤6: 生成{split}集订单...")
        orders_list = self.generate_orders(
            daily_heatmaps, selected_dates, node_coords, grid_to_node, node_node, nrows, ncols,
            station_mask
        )
        print(f"  生成了 {len(orders_list)} 天的订单")
        
        # 8. 处理订单数据并分配时间
        print("步骤7: 处理订单数据并分配时间...")
        instances = []
        frame_accumulate_list = []
        
        for day_idx, orders in enumerate(orders_list):
            if len(orders) == 0:
                continue
            
            # 分配出现时间
            appear_times = self.assign_order_times(len(orders))
            
            # 构建订单数组
            orders_array = np.array(orders)
            requests_from = orders_array[:, 0].astype(np.int64)
            requests_to = orders_array[:, 1].astype(np.int64)
            
            # 排序订单按出现时间并截断/填充到n_tot_requests长度
            sort_idx = np.argsort(appear_times)
            requests_from = requests_from[sort_idx]
            requests_to = requests_to[sort_idx]
            appear_times = appear_times[sort_idx]
            
            n_actual = len(requests_from)
            if n_actual < self.n_tot_requests:
                padding_len = self.n_tot_requests - n_actual
                requests_from = np.pad(requests_from, (0, padding_len), constant_values=0)
                requests_to = np.pad(requests_to, (0, padding_len), constant_values=0)
                appear_times = np.pad(appear_times, (0, padding_len), constant_values=self.n_frame)
            else:
                requests_from = requests_from[:self.n_tot_requests]
                requests_to = requests_to[:self.n_tot_requests]
                appear_times = appear_times[:self.n_tot_requests]
            
            instances.append({
                'from': requests_from,
                'to': requests_to,
                'appear': appear_times
            })
            
            # 计算累积请求数
            frame_counts = np.bincount(appear_times, minlength=self.n_frame + 1)
            frame_accumulate = np.cumsum(frame_counts)
            frame_accumulate_list.append(frame_accumulate)
        
        if len(instances) == 0:
            raise ValueError(f"No {split} instances generated")
        
        # 9. 转换为tensor
        print("步骤8: 转换为tensor...")
        n_inst = len(instances)
        
        requests_from = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        requests_to = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        requests_value = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        frame_accumulate_tensor = torch.zeros(n_inst, self.n_frame + 1, dtype=torch.int64, device=self.device)
        
        for i, inst in enumerate(instances):
            requests_from[i] = torch.from_numpy(inst['from']).to(self.device)
            requests_to[i] = torch.from_numpy(inst['to']).to(self.device)
            frame_accumulate_tensor[i] = torch.from_numpy(frame_accumulate_list[i]).to(self.device)
        
        # 计算订单value（等于起点和终点之间的曼哈顿距离）
        node_node_tensor = torch.from_numpy(node_node).to(self.device).long()
        from_valid = (requests_from < self.n_node) & (requests_to < self.n_node)
        requests_value = node_node_tensor[
            requests_from.clamp(0, self.n_node-1), 
            requests_to.clamp(0, self.n_node-1)
        ].long() * from_valid.long()
        
        # 10. 构建保存数据
        data = {
            "coord": torch.from_numpy(node_coords).to(self.device),
            "node_node": torch.from_numpy(node_node).to(self.device),
            "station_idx": torch.from_numpy(station_indices).to(self.device),
            "station_mask": torch.from_numpy(station_mask).to(self.device),
            "station_node_node": torch.from_numpy(station_node_node).to(self.device),
            "requests_from": requests_from,
            "requests_to": requests_to,
            "requests_value": requests_value,
            "frame_accumulate": frame_accumulate_tensor,
            "metadata": {
                "city": self.city,
                "n_node": self.n_node,
                "n_station": self.n_station,
                "n_tot_requests": self.n_tot_requests,
                "min_direct_dist": self.min_direct_dist,
                "n_frame": self.n_frame,
                "n_requested_frame": self.n_requested_frame,
                "n_init_requests": self.n_init_requests,
                "n_norm_requests": self.n_norm_requests,
                "n_instances": n_inst
            }
        }
        
        if split == 'train':
            data["metadata"]["train_days"] = self.train_days
        else:
            data["metadata"]["test_days"] = self.test_days
        
        print(f"数据生成完成！{split}集实例数: {n_inst}")
        
        return data


def generate_filename(city, split, n_node, n_station, n_tot_requests, min_direct_dist):
    """生成数据集文件名"""
    return f"{city}_{split}_node{n_node}_station{n_station}_totreq{n_tot_requests}_mindist{min_direct_dist}.pt"


def output_nodes_info(data, city, split, save_dir):
    """输出所有节点的信息（节点序号、坐标、是否为station）"""
    # 获取数据
    node_coords = data["coord"].cpu().numpy()  # (n_node, 2) where 2 is (lng, lat)
    station_mask = data["station_mask"].cpu().numpy()  # (n_node,)
    metadata = data["metadata"]
    
    n_node = metadata["n_node"]
    n_pickup = int(n_node * PICKUP_NODE_RATIO)
    
    # 创建输出目录
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 准备输出数据
    nodes_info = []
    for i in range(n_node):
        lng, lat = node_coords[i]
        is_station = bool(station_mask[i])
        node_type = "station" if is_station else ("pickup" if i < n_pickup else "delivery")
        nodes_info.append({
            "node_id": i,
            "longitude": lng,
            "latitude": lat,
            "is_station": is_station,
            "node_type": node_type
        })
    
    # 按经纬度排序（先按纬度升序，再按经度升序，相同纬度的节点会挨在一起）
    def sort_key(info):
        return (info['latitude'], info['longitude'])  # 升序排序
    
    # 保存到CSV文件（按经纬度排序）
    df = pd.DataFrame(nodes_info)
    df_sorted = df.sort_values(by=['latitude', 'longitude'], ascending=[True, True])
    filename = f"nodes_info_{city}_{split}_n_node{n_node}.csv"
    csv_path = save_dir / filename
    df_sorted.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n节点信息已保存到: {csv_path}")
    
    # 输出到控制台
    print(f"\n{'='*80}")
    print(f"节点信息汇总 ({city.upper()} - {split.upper()})")
    print(f"{'='*80}")
    print(f"总节点数: {n_node}")
    print(f"  - Pickup节点: {n_pickup}")
    print(f"  - Delivery节点: {n_node - n_pickup}")
    print(f"  - Station节点: {station_mask.sum()}")
    print(f"\n详细节点信息（按纬度升序、经度升序排序，相同纬度节点相邻）:")
    print(f"{'节点ID':<8} {'经度':<15} {'纬度':<15} {'类型':<12} {'是否Station':<12}")
    print(f"{'-'*80}")
    
    # 所有节点一起排序输出（不区分station和非station）
    sorted_nodes = sorted(nodes_info, key=sort_key)
    for info in sorted_nodes:
        station_label = '是' if info['is_station'] else '否'
        print(f"{info['node_id']:<8} {info['longitude']:<15.8f} {info['latitude']:<15.8f} "
              f"{info['node_type']:<12} {station_label:<12}")
    
    print(f"{'='*80}\n")
    
    return csv_path


def plot_nodes_and_stations(data, city, split, save_dir):
    """绘制节点和起降站分布图"""
    # 获取数据
    node_coords = data["coord"].cpu().numpy()  # (n_node, 2) where 2 is (lng, lat)
    station_mask = data["station_mask"].cpu().numpy()  # (n_node,)
    metadata = data["metadata"]
    
    n_node = metadata["n_node"]
    n_pickup = int(n_node * PICKUP_NODE_RATIO)
    
    # 获取城市配置
    config = CITY_CONFIGS[city]
    lng_range = config["lng_range"]
    lat_range = config["lat_range"]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制城市边界框
    rect = plt.Rectangle(
        (lng_range[0], lat_range[0]),
        lng_range[1] - lng_range[0],
        lat_range[1] - lat_range[0],
        linewidth=2, edgecolor='gray', facecolor='lightgray', alpha=0.3, label='City Boundary'
    )
    ax.add_patch(rect)
    
    # 分类节点
    pickup_nodes = []
    delivery_nodes = []
    station_nodes = []
    
    for i in range(n_node):
        lng, lat = node_coords[i]
        if station_mask[i]:
            station_nodes.append((lng, lat))
        elif i < n_pickup:
            pickup_nodes.append((lng, lat))
        else:
            delivery_nodes.append((lng, lat))
    
    # 绘制pickup节点
    if pickup_nodes:
        pickup_lngs, pickup_lats = zip(*pickup_nodes)
        ax.scatter(pickup_lngs, pickup_lats, c='blue', marker='o', s=100, 
                  alpha=0.7, label=f'Pickup Nodes ({len(pickup_nodes)})', edgecolors='darkblue', linewidths=1.5)
    
    # 绘制delivery节点
    if delivery_nodes:
        delivery_lngs, delivery_lats = zip(*delivery_nodes)
        ax.scatter(delivery_lngs, delivery_lats, c='green', marker='s', s=100, 
                  alpha=0.7, label=f'Delivery Nodes ({len(delivery_nodes)})', edgecolors='darkgreen', linewidths=1.5)
    
    # 绘制站点
    if station_nodes:
        station_lngs, station_lats = zip(*station_nodes)
        ax.scatter(station_lngs, station_lats, c='red', marker='^', s=200, 
                  alpha=0.9, label=f'Stations ({len(station_nodes)})', edgecolors='darkred', linewidths=2)
    
    # 设置坐标轴
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'{city.upper()} City Node Distribution - {split.upper()} Set', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # 设置坐标轴范围（稍微扩展一点以便查看）
    lng_margin = (lng_range[1] - lng_range[0]) * 0.05
    lat_margin = (lat_range[1] - lat_range[0]) * 0.05
    ax.set_xlim(lng_range[0] - lng_margin, lng_range[1] + lng_margin)
    ax.set_ylim(lat_range[0] - lat_margin, lat_range[1] + lat_margin)
    
    plt.tight_layout()
    
    # 保存图片
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_station = len(station_nodes)  # 获取站点数量
    filename = f"nodes_stations_{city}_{split}_n_node{n_node}_n_station{n_station}.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Node and station distribution plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='生成LADE数据集')
    # 默认yaml文件路径
    default_yaml_path = '/mnt/jfs6/g-bairui/results/data/env_lade.yaml'
    parser.add_argument('--yaml', type=str, default=str(default_yaml_path), 
                       help=f'YAML配置文件路径（默认: {default_yaml_path}）')
    parser.add_argument('--city', type=str, choices=['hz', 'cq', 'sh'], help='城市（如果yaml中未指定，默认为hz）')
    parser.add_argument('--n_node', type=int, help='节点数量')
    parser.add_argument('--n_station', type=int, help='站点数量')
    parser.add_argument('--n_tot_requests', type=int, help='总订单数')
    parser.add_argument('--min_direct_dist', type=int, help='订单最短距离')
    parser.add_argument('--n_frame', type=int, help='总时间步数')
    parser.add_argument('--n_requested_frame', type=int, help='订单请求时间范围')
    parser.add_argument('--n_init_requests', type=int, help='初始订单数')
    parser.add_argument('--n_norm_requests', type=int, help='正常订单数')
    parser.add_argument('--train_days', type=int, default=60, help='训练集天数')
    parser.add_argument('--test_days', type=int, default=20, help='测试集天数')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='随机种子')
    parser.add_argument('--seed_env', type=int, default=DEFAULT_SEED_ENV, help='环境随机种子')
    parser.add_argument('--output_dir', type=str, default='/mnt/jfs6/g-bairui/results/data/LADE/processed', help='输出目录')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'both'], default='both', 
                       help='生成训练集、测试集或两者')
    parser.add_argument('--draw', action='store_true', 
                       help='是否绘制节点和起降站分布图（默认不绘制）')
    
    args = parser.parse_args()
    
    # 从yaml文件读取配置（如果文件存在）
    config = {}
    yaml_path = Path(args.yaml)
    if yaml_path.exists():
        print(f"从YAML文件读取配置: {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.load(f, yaml.Loader)
            config.update(yaml_config)
    else:
        print(f"警告: YAML文件不存在: {yaml_path}，将仅使用命令行参数")
    
    # 命令行参数覆盖yaml配置
    # city参数：命令行参数优先，如果没有则使用yaml中的值，最后使用默认值'hz'
    if args.city:
        config['city'] = args.city
    elif 'city' not in config:
        config['city'] = 'hz'  # 默认值
    if args.n_node:
        config['n_node'] = args.n_node
    if args.n_station:
        config['n_station'] = args.n_station
    if args.n_tot_requests:
        config['n_tot_requests'] = args.n_tot_requests
    elif 'n_init_requests' in config and 'n_norm_requests' in config:
        config['n_tot_requests'] = config['n_init_requests'] + config['n_norm_requests']
    if args.min_direct_dist:
        config['min_direct_dist'] = args.min_direct_dist
    if args.n_frame:
        config['n_frame'] = args.n_frame
    if args.n_requested_frame:
        config['n_requested_frame'] = args.n_requested_frame
    if args.n_init_requests:
        config['n_init_requests'] = args.n_init_requests
    if args.n_norm_requests:
        config['n_norm_requests'] = args.n_norm_requests
    if args.train_days:
        config['train_days'] = args.train_days
    if args.test_days:
        config['test_days'] = args.test_days
    
    # 检查必需参数
    required_params = ['city', 'n_node', 'n_station', 'n_tot_requests', 'min_direct_dist',
                      'n_frame', 'n_requested_frame', 'n_init_requests', 'n_norm_requests']
    missing_params = [p for p in required_params if p not in config]
    if missing_params:
        raise ValueError(
            f"缺少必需参数: {missing_params}\n"
            f"请通过以下方式提供参数：\n"
            f"1. 在YAML文件中设置（推荐）\n"
            f"2. 通过命令行参数提供（如 --n_node 30）\n"
            f"3. 当前YAML文件路径: {yaml_path}"
        )
    
    # 创建输出目录
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_node = 40
    n_station = 6
    n_tot_requests = 220
    n_init_requests = 20
    n_norm_requests = 200
    seed = 1
    seed_env = args.seed_env
    min_direct_dist = 10
    n_frame = 80
    
    config['n_node'] = n_node
    config['n_station'] = n_station
    config['n_tot_requests'] = n_tot_requests
    config['n_init_requests'] = n_init_requests
    config['n_norm_requests'] = n_norm_requests
    config['seed'] = seed
    config['seed_env'] = seed_env
    config['min_direct_dist'] = min_direct_dist
    config['n_frame'] = n_frame
    
    print(f"n_node: {n_node}")
    print(f"n_station: {n_station}")
    print(f"n_tot_requests: {n_tot_requests}")
    print(f"n_init_requests: {n_init_requests}")
    print(f"n_norm_requests: {n_norm_requests}")
    print(f"seed: {seed}")
    print(f"seed_env: {seed_env}")
    print(f"n_frame: {n_frame}")
    print(f"min_direct_dist: {min_direct_dist}")
    # 创建生成器
    generator = LADEDataGenerator(
        city=config['city'],
        n_node=config['n_node'],
        n_station=config['n_station'],
        n_tot_requests=config['n_tot_requests'],
        min_direct_dist=config['min_direct_dist'],
        n_frame=config['n_frame'],
        n_requested_frame=config['n_requested_frame'],
        n_init_requests=config['n_init_requests'],
        n_norm_requests=config['n_norm_requests'],
        train_days=config.get('train_days', 60),
        test_days=config.get('test_days', 20),
        # seed=args.seed,
        # seed_env=args.seed_env
        seed=seed,
        seed_env=seed_env
    )
    
    # 生成数据集
    splits_to_generate = []
    if args.split in ['train', 'both']:
        splits_to_generate.append('train')
    if args.split in ['test', 'both']:
        splits_to_generate.append('test')
    
    for split in splits_to_generate:
        print(f"\n{'='*60}")
        print(f"生成{split}集")
        print(f"{'='*60}")
        data = generator.generate_dataset(split=split)
        
        # 保存数据
        filename = generate_filename(
            config['city'], split, config['n_node'], config['n_station'],
            config['n_tot_requests'], config['min_direct_dist']
        )
        output_path = output_dir / filename
        
        print(f"保存数据到: {output_path}")
        torch.save(data, output_path)
        print(f"保存完成！")
        
        # 输出节点信息（包含station）
        print(f"\n输出{split}集节点信息...")
        info_dir = output_dir / 'nodes_info'
        output_nodes_info(data, config['city'], split, info_dir)
        
        # 如果启用绘图，绘制节点和起降站分布图
        if args.draw:
            print(f"绘制{split}集节点和起降站分布图...")
            images_dir = output_dir / 'images'
            plot_nodes_and_stations(data, config['city'], split, images_dir)


if __name__ == '__main__':
    main()

