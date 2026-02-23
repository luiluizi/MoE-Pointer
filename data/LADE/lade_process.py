"""
LADE dataset preprocessing script
Generate training and test sets from CSV data and save them as .pt files
"""
import torch
import pandas as pd
import numpy as np
import argparse
import yaml
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add the project root directory to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import helper functions
from envs.env_lade import (
    CITY_CONFIGS, meters_to_degrees, build_grid, assign_points_to_grid
)

# Fixed parameters
GRID_SIZE_M = 1000.0
PICKUP_NODE_RATIO = 0.4
DEFAULT_SEED = 0
DEFAULT_SEED_ENV = 1


class LADEDataGenerator:
    """LADE Dataset Generator"""
    
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
        """Load pickup and delivery CSV data"""
        config = CITY_CONFIGS[self.city]
        base_dir = Path("./data/LADE")
        
        # pickup_path = base_dir / config["pickup_csv"]
        # delivery_path = base_dir / config["delivery_csv"]
        pickup_path = config["pickup_csv"]
        delivery_path = config["delivery_csv"]
        print(pickup_path)
        
        pickup_df = pd.read_csv(pickup_path)
        delivery_df = pd.read_csv(delivery_path)
        
        required_pickup_cols = ['pickup_gps_lng', 'pickup_gps_lat']
        required_delivery_cols = ['delivery_gps_lng', 'delivery_gps_lat']
        
        missing_pickup = [c for c in required_pickup_cols if c not in pickup_df.columns]
        missing_delivery = [c for c in required_delivery_cols if c not in delivery_df.columns]
        
        if missing_pickup:
            raise ValueError(f"Pickup CSV is missing required fields: {missing_pickup}. Available fields: {pickup_df.columns.tolist()}")
        if missing_delivery:
            raise ValueError(f"Delivery CSV is missing required fields: {missing_delivery}. Available fields: {delivery_df.columns.tolist()}")

        # Extract required columns (including the ds field if it exists)
        pickup_cols = required_pickup_cols + (['ds'] if 'ds' in pickup_df.columns else [])
        delivery_cols = required_delivery_cols + (['ds'] if 'ds' in delivery_df.columns else [])
        
        pickup_df = pickup_df[pickup_cols].copy()
        delivery_df = delivery_df[delivery_cols].copy()
        
        # Convert to numeric type and handle missing values
        pickup_df['pickup_gps_lng'] = pd.to_numeric(pickup_df['pickup_gps_lng'], errors='coerce')
        pickup_df['pickup_gps_lat'] = pd.to_numeric(pickup_df['pickup_gps_lat'], errors='coerce')
        delivery_df['delivery_gps_lng'] = pd.to_numeric(delivery_df['delivery_gps_lng'], errors='coerce')
        delivery_df['delivery_gps_lat'] = pd.to_numeric(delivery_df['delivery_gps_lat'], errors='coerce')
        
        # Drop missing values
        pickup_df = pickup_df.dropna(subset=['pickup_gps_lng', 'pickup_gps_lat'])
        delivery_df = delivery_df.dropna(subset=['delivery_gps_lng', 'delivery_gps_lat'])
        
        # Unify column names
        pickup_df = pickup_df.rename(columns={'pickup_gps_lng': 'lng', 'pickup_gps_lat': 'lat'})
        delivery_df = delivery_df.rename(columns={'delivery_gps_lng': 'lng', 'delivery_gps_lat': 'lat'})
        
        # Process date field (ds field, format example: 0618, 0814, 0901)
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
        
        # Filter data range
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
        """Compute daily heatmaps, grouped by date (ds field)"""
        # Calculate grid parameters
        center_lat = (lat_min + lat_max) / 2.0
        delta_lng, delta_lat = meters_to_degrees(self.grid_size_m, center_lat)
        lng_edges, lat_edges, ncols, nrows = build_grid(lng_min, lng_max, lat_min, lat_max, delta_lng, delta_lat)
        
        # Calculate grid center coordinates
        grid_centers = []
        for r in range(nrows):
            for c in range(ncols):
                lat_c = (lat_edges[r] + lat_edges[r+1]) / 2.0
                lng_c = (lng_edges[c] + lng_edges[c+1]) / 2.0
                grid_idx = r * ncols + c
                grid_centers.append((lng_c, lat_c, r, c, grid_idx))
        
        daily_heatmaps = {}
        
        # Check if date information exists
        pickup_has_date = 'ds' in pickup_df.columns and pickup_df['ds'].notna().any()
        delivery_has_date = 'ds' in delivery_df.columns and delivery_df['ds'].notna().any()
        
        if pickup_has_date or delivery_has_date:
            # Get all unique dates and sort them
            all_dates = set()
            if pickup_has_date:
                all_dates.update(pickup_df['ds'].dropna().unique())
            if delivery_has_date:
                all_dates.update(delivery_df['ds'].dropna().unique())
            all_dates = sorted(list(all_dates))
            
            # Compute heatmaps grouped by date
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
            # If no date information exists, treat all data as the same day
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
        # Cumulative heatmap
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
        
        # Calculate number of nodes
        n_pickup_nodes = int(self.n_node * self.pickup_node_ratio)
        n_delivery_nodes = self.n_node - n_pickup_nodes
        
        # Get grid center coordinates
        grid_info = daily_heatmaps[selected_dates[0]]
        grid_centers = grid_info['grid_centers']
        nrows = grid_info['nrows']
        ncols = grid_info['ncols']
        
        # Select pickup nodes (sorted by heat intensity)
        pickup_heatmap_flat = cumulative_pickup.flatten()
        top_pickup_indices_flat = np.argsort(pickup_heatmap_flat)[-n_pickup_nodes:]
        
        # Select delivery nodes (excluding selected pickup nodes)
        delivery_heatmap_flat = cumulative_delivery.flatten()
        available_indices = [i for i in range(len(delivery_heatmap_flat)) 
                           if i not in top_pickup_indices_flat]
        if len(available_indices) < n_delivery_nodes:
            top_delivery_indices_flat = np.argsort(delivery_heatmap_flat)[-n_delivery_nodes:]
        else:
            available_heatmap_vals = [delivery_heatmap_flat[i] for i in available_indices]
            top_available_indices = np.argsort(available_heatmap_vals)[-n_delivery_nodes:]
            top_delivery_indices_flat = [available_indices[i] for i in top_available_indices]
        
        # Convert to grid coordinates
        pickup_nodes = [grid_centers[idx] for idx in top_pickup_indices_flat if idx < len(grid_centers)]
        delivery_nodes = [grid_centers[idx] for idx in top_delivery_indices_flat if idx < len(grid_centers)]
        
        # Merge nodes and extract coordinates
        all_nodes = pickup_nodes + delivery_nodes
        node_coords = np.array([(node[0], node[1]) for node in all_nodes])
        
        # Create node index mapping (grid(r,c) -> node index)
        grid_to_node = {(node[2], node[3]): idx for idx, node in enumerate(all_nodes) if len(node) >= 4}
        
        return node_coords, grid_to_node, nrows, ncols
    
    def compute_distance_matrix(self, node_coords):
        """Compute Manhattan distance matrix based on node coordinates"""
        # Calculate Manhattan distance (longitude/latitude differences)
        lng_diff = np.abs(node_coords[:, 0:1] - node_coords[:, 0])
        lat_diff = np.abs(node_coords[:, 1:2] - node_coords[:, 1])
        manhattan_dist = lng_diff + lat_diff
        
        # Convert to integer distance in 1km units
        node_node = (manhattan_dist * 111.32 + 0.5).astype(int)
        np.fill_diagonal(node_node, 0)
        
        return node_node
    
    def select_stations(self, node_coords, manual_station_indices=None):
        """Select n_station nodes as stations from all nodes
        
        Args:
            node_coords: Array of node coordinates
            manual_station_indices: List of manually specified station indices. If provided, use these; otherwise, select randomly
        """
        n_node = len(node_coords)
        if self.n_station > n_node:
            raise ValueError(f"n_station ({self.n_station}) > n_node ({n_node})")
        
        # Use manually specified station indices if provided
        if manual_station_indices is not None:
            station_indices = np.array(manual_station_indices, dtype=np.int64)
            
            # Validate index validity
            if len(station_indices) != self.n_station:
                raise ValueError(
                    f"Number of manually specified stations ({len(station_indices)}) does not match n_station ({self.n_station})"
                )
            if np.any(station_indices < 0) or np.any(station_indices >= n_node):
                raise ValueError(
                    f"Station indices must be in the range [0, {n_node-1}], but found: {station_indices}"
                )
            if len(np.unique(station_indices)) != len(station_indices):
                raise ValueError(f"Station indices cannot be duplicated: {station_indices}")
            
            station_indices = np.sort(station_indices)
            print(f"  Using manually specified station indices: {station_indices.tolist()}")
        else:
            # Otherwise select randomly
            perm = torch.randperm(n_node, generator=self.rng_env, device=self.device)
            station_indices = perm[:self.n_station].cpu().numpy()
            station_indices = np.sort(station_indices)
            print(f"  Randomly selected station indices: {station_indices.tolist()}")
        # Add sentinel point
        station_indices = np.concatenate([station_indices, [n_node]])
        
        return station_indices
    
    def generate_orders(self, daily_heatmaps, selected_dates, node_coords, grid_to_node, 
                       node_node, nrows, ncols, station_mask):
        """Sample orders based on daily heatmaps"""
        all_orders = []
        
        # Calculate available nodes (all non-station nodes can be used as start or end points)
        available_nodes = np.where(~station_mask)[0]
        
        if len(available_nodes) < 2:
            raise ValueError(f"Not enough available nodes (non-station): {len(available_nodes)}, need at least 2")
        
        for date in selected_dates:
            if date not in daily_heatmaps:
                continue
                
            heatmap_info = daily_heatmaps[date]
            pickup_heatmap = heatmap_info['pickup']
            delivery_heatmap = heatmap_info['delivery']
            
            # Map heatmaps to probabilities for all nodes
            pickup_node_probs = np.zeros(self.n_node)
            delivery_node_probs = np.zeros(self.n_node)
            
            # Calculate total grid heat intensity for each node
            for r in range(nrows):
                for c in range(ncols):
                    if (r, c) in grid_to_node:
                        node_idx = grid_to_node[(r, c)]
                        pickup_node_probs[node_idx] += pickup_heatmap[r, c]
                        delivery_node_probs[node_idx] += delivery_heatmap[r, c]
            
            # Normalize probabilities (only consider non-station nodes)
            for probs in [pickup_node_probs, delivery_node_probs]:
                probs_available = probs[available_nodes]
                if probs_available.sum() > 0:
                    probs[available_nodes] = probs_available / probs_available.sum()
                else:
                    probs[available_nodes] = 1.0 / len(available_nodes)
                probs[station_mask] = 0
            
            # generate requests
            orders_for_date = []
            valid_pairs = [(i, j) for i in available_nodes for j in available_nodes 
                          if i != j and node_node[i, j] >= self.min_direct_dist]
            
            if len(valid_pairs) == 0:
                print(f"Warning: No valid node pairs found for date {date}, using relaxed distance")
                min_dist = max(1, self.min_direct_dist - 3)
                valid_pairs = [(i, j) for i in available_nodes for j in available_nodes 
                              if i != j and node_node[i, j] >= min_dist]
            
            if len(valid_pairs) == 0:
                valid_pairs = [(i, j) for i in available_nodes for j in available_nodes if i != j]
            
            # Sample node pairs by probability
            if len(valid_pairs) > 0:
                pair_probs = np.array([
                    pickup_node_probs[pair[0]] * delivery_node_probs[pair[1]]
                    for pair in valid_pairs
                ])
                if pair_probs.sum() > 0:
                    pair_probs = pair_probs / pair_probs.sum()
                else:
                    pair_probs = np.ones(len(valid_pairs)) / len(valid_pairs)
                
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
        print(f"Start generating LADE dataset (city={self.city}, split={split})...")
        self.rng.manual_seed(self.seed)
        self.rng_env.manual_seed(self.seed_env)
        
        # 1. Load data
        pickup_df, delivery_df, lng_min, lng_max, lat_min, lat_max = self.load_data()
        print(f"  Pickup data: {len(pickup_df)} records")
        print(f"  Delivery data: {len(delivery_df)} records")
        
        # 2. Compute daily heatmaps
        daily_heatmaps = self.compute_daily_heatmaps(pickup_df, delivery_df, lng_min, lng_max, lat_min, lat_max)
        print(f"  Computed heatmaps for {len(daily_heatmaps)} days")
        
        # 3. Determine date range
        all_dates = sorted(daily_heatmaps.keys())
        n_total_days = len(all_dates)
        
        if n_total_days == 0:
            raise ValueError("No daily heatmaps computed")
        
        # Split dates for training and test sets
        if n_total_days == 1:
            selected_dates = all_dates
        else:
            if split == 'train':
                selected_dates = all_dates[:min(self.train_days, n_total_days)]
            else:
                train_start = min(self.train_days, n_total_days)
                test_end = min(train_start + self.test_days, n_total_days)
                selected_dates = all_dates[train_start:test_end]
        print(f"  Using {len(selected_dates)} days of data")
        # 4. Select nodes (use cumulative heatmaps of all dates to ensure consistent nodes for train/test)
        node_coords, grid_to_node, nrows, ncols = self.select_nodes(daily_heatmaps, all_dates)
        print(f"  Selected {len(node_coords)} nodes")
        
        # 5. Compute distance matrix
        node_node = self.compute_distance_matrix(node_coords)
        print(f"  Distance matrix range: [{node_node.min()}, {node_node.max()}]")
        
        # 6. Select stations (reset rng_env to ensure consistent random state for train/test)
        # Read station_id from CITY_CONFIGS (if exists)
        config = CITY_CONFIGS[self.city]
        manual_station_indices = None
        if 'station_id' in config and config['station_id'] is not None:
            manual_station_indices = config['station_id']
            if len(manual_station_indices) != self.n_station:
                raise ValueError(
                    f"Number of station_ids in CITY_CONFIGS ({len(manual_station_indices)}) does not match n_station ({self.n_station})"
                )
            print(f"  Loaded manually specified station indices from CITY_CONFIGS: {manual_station_indices}")
        else:
            self.rng_env.manual_seed(self.seed_env)
            print(f"seed_env: {self.seed_env}")
        
        station_indices = self.select_stations(node_coords, manual_station_indices=manual_station_indices)
        station_mask = np.zeros(self.n_node, dtype=bool)
        station_mask[station_indices[:-1]] = True
        station_node_node = node_node[np.ix_(station_indices[:-1], station_indices[:-1])]
        print(f"  Selected {len(station_indices)-1} stations")
        
        # 7. Generate orders
        orders_list = self.generate_orders(
            daily_heatmaps, selected_dates, node_coords, grid_to_node, node_node, nrows, ncols,
            station_mask
        )
        print(f"  Generated orders for {len(orders_list)} days")
        
        # 8. Process order data and assign timestamps
        instances = []
        frame_accumulate_list = []
        
        for day_idx, orders in enumerate(orders_list):
            if len(orders) == 0:
                continue
            appear_times = self.assign_order_times(len(orders))
            # Construct order array
            orders_array = np.array(orders)
            requests_from = orders_array[:, 0].astype(np.int64)
            requests_to = orders_array[:, 1].astype(np.int64)
            # Sort orders by appearance time and truncate/pad to n_tot_requests length
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
            
            frame_counts = np.bincount(appear_times, minlength=self.n_frame + 1)
            frame_accumulate = np.cumsum(frame_counts)
            frame_accumulate_list.append(frame_accumulate)
        
        if len(instances) == 0:
            raise ValueError(f"No {split} instances generated")
        
        # 9. Convert to tensor
        n_inst = len(instances)
        requests_from = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        requests_to = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        requests_value = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        frame_accumulate_tensor = torch.zeros(n_inst, self.n_frame + 1, dtype=torch.int64, device=self.device)
        
        for i, inst in enumerate(instances):
            requests_from[i] = torch.from_numpy(inst['from']).to(self.device)
            requests_to[i] = torch.from_numpy(inst['to']).to(self.device)
            frame_accumulate_tensor[i] = torch.from_numpy(frame_accumulate_list[i]).to(self.device)
        
        # Calculate order values (equal to Manhattan distance between start and end points)
        node_node_tensor = torch.from_numpy(node_node).to(self.device).long()
        from_valid = (requests_from < self.n_node) & (requests_to < self.n_node)
        requests_value = node_node_tensor[
            requests_from.clamp(0, self.n_node-1), 
            requests_to.clamp(0, self.n_node-1)
        ].long() * from_valid.long()
        
        # 10. Construct and save data
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
        print(f"Dataset generation completed! Number of {split} set instances: {n_inst}")
        return data

def generate_filename(city, split, n_node, n_station, n_tot_requests, min_direct_dist):
    return f"{city}_{split}_node{n_node}_station{n_station}_totreq{n_tot_requests}_mindist{min_direct_dist}.pt"

def plot_nodes_and_stations(data, city, split, save_dir):
    node_coords = data["coord"].cpu().numpy()  # (n_node, 2) where 2 is (lng, lat)
    station_mask = data["station_mask"].cpu().numpy()  # (n_node,)
    metadata = data["metadata"]
    
    n_node = metadata["n_node"]
    n_pickup = int(n_node * PICKUP_NODE_RATIO)

    config = CITY_CONFIGS[city]
    lng_range = config["lng_range"]
    lat_range = config["lat_range"]
    fig, ax = plt.subplots(figsize=(12, 10))

    rect = plt.Rectangle(
        (lng_range[0], lat_range[0]),
        lng_range[1] - lng_range[0],
        lat_range[1] - lat_range[0],
        linewidth=2, edgecolor='gray', facecolor='lightgray', alpha=0.3, label='City Boundary'
    )
    ax.add_patch(rect)
    
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
  
    if pickup_nodes:
        pickup_lngs, pickup_lats = zip(*pickup_nodes)
        ax.scatter(pickup_lngs, pickup_lats, c='blue', marker='o', s=100, 
                  alpha=0.7, label=f'Pickup Nodes ({len(pickup_nodes)})', edgecolors='darkblue', linewidths=1.5)
    if delivery_nodes:
        delivery_lngs, delivery_lats = zip(*delivery_nodes)
        ax.scatter(delivery_lngs, delivery_lats, c='green', marker='s', s=100, 
                  alpha=0.7, label=f'Delivery Nodes ({len(delivery_nodes)})', edgecolors='darkgreen', linewidths=1.5)
    if station_nodes:
        station_lngs, station_lats = zip(*station_nodes)
        ax.scatter(station_lngs, station_lats, c='red', marker='^', s=200, 
                  alpha=0.9, label=f'Stations ({len(station_nodes)})', edgecolors='darkred', linewidths=2)
    
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'{city.upper()} City Node Distribution - {split.upper()} Set', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    lng_margin = (lng_range[1] - lng_range[0]) * 0.05
    lat_margin = (lat_range[1] - lat_range[0]) * 0.05
    ax.set_xlim(lng_range[0] - lng_margin, lng_range[1] + lng_margin)
    ax.set_ylim(lat_range[0] - lat_margin, lat_range[1] + lat_margin)
    
    plt.tight_layout()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_station = len(station_nodes) 
    filename = f"nodes_stations_{city}_{split}_n_node{n_node}_n_station{n_station}.png"
    save_path = save_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Node and station distribution plot saved to: {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate LADE dataset')
    default_yaml_path = '../../envs/config/lade.yaml'
    parser.add_argument('--yaml', type=str, default=str(default_yaml_path), 
                       help=f'YAML configuration file path (default: {default_yaml_path}）')
    parser.add_argument('--city', type=str, choices=['hz', 'cq', 'sh'], default='hz')
    parser.add_argument('--n_node', type=int, help='Number of nodes')
    parser.add_argument('--n_station', type=int, help='Number of stations')
    parser.add_argument('--n_tot_requests', type=int, help='Total number of requests')
    parser.add_argument('--min_direct_dist', type=int, help='Minimum direct distance for requests')
    parser.add_argument('--n_frame', type=int, help='Total number of time steps')
    parser.add_argument('--n_requested_frame', type=int, help='Time range for requests')
    parser.add_argument('--n_init_requests', type=int, help='Number of initial requests')
    parser.add_argument('--n_norm_requests', type=int, help='Number of normal requests')
    parser.add_argument('--train_days', type=int, default=60, help='Number of days for training set')
    parser.add_argument('--test_days', type=int, default=20, help='Number of days for test set')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='Random seed')
    parser.add_argument('--seed_env', type=int, default=DEFAULT_SEED_ENV, help='Environment random seed')
    parser.add_argument('--output_dir', type=str, default='./processed', help='Output directory')
    parser.add_argument('--split', type=str, choices=['train', 'test', 'both'], default='both', 
                    help='Generate training set, test set, or both')
    parser.add_argument('--draw', action='store_true', 
                    help='Whether to plot distribution of nodes and stations (not plotted by default)')
    args = parser.parse_args()
    
    config = {}
    yaml_path = Path(args.yaml)
    if yaml_path.exists():
        print(f"Loading configuration from YAML file: {yaml_path}")
        with open(yaml_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.load(f, yaml.Loader)
            config.update(yaml_config)
    else:
        print(f"Warning: YAML file does not exist: {yaml_path}, only command-line arguments will be used")
    
    # Command-line arguments override YAML configuration
    # city parameter: command-line arguments take precedence, then values from YAML, and finally the default value 'hz'
    if args.city:
        config['city'] = args.city
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
    
    required_params = ['city', 'n_node', 'n_station', 'n_tot_requests', 'min_direct_dist',
                      'n_frame', 'n_requested_frame', 'n_init_requests', 'n_norm_requests']
    missing_params = [p for p in required_params if p not in config]
    if missing_params:
        raise ValueError(
            f"Missing required parameters: {missing_params}\n"
            f"Please provide the parameters in one of the following ways:\n"
            f"1. Set in the YAML file (recommended)\n"
            f"2. Provide via command-line arguments (e.g., --n_node 30)\n"
            f"3. Current YAML file path: {yaml_path}"
        )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        seed=config['seed'],
        seed_env=config['seed_env'],
    )
    
    # Generate dataset
    splits_to_generate = []
    if args.split in ['train', 'both']:
        splits_to_generate.append('train')
    if args.split in ['test', 'both']:
        splits_to_generate.append('test')
    
    for split in splits_to_generate:
        print(f"\n{'='*60}")
        print(f"Generating {split} set")
        print(f"{'='*60}")
        data = generator.generate_dataset(split=split)
        
        # 保存数据
        filename = generate_filename(
            config['city'], split, config['n_node'], config['n_station'],
            config['n_tot_requests'], config['min_direct_dist']
        )
        output_path = output_dir / filename
        
        print(f"Saving data to: {output_path}")
        torch.save(data, output_path)
        print(f"Saved successfully!")
        
        # Plot distribution of nodes and stations if drawing is enabled
        if args.draw:
            print(f"Plotting distribution of nodes and stations for {split} set...")
            images_dir = output_dir / 'images'
            plot_nodes_and_stations(data, config['city'], split, images_dir)


if __name__ == '__main__':
    main()

