import torch
import pandas as pd
import numpy as np
import geohash
import matplotlib.pyplot as plt

from .env import DroneTransferEnv, RELATION

occupancy_logged = False
obs_func_map = {}
trans_func_map = {}

class DroneTransferEnvDHRD(DroneTransferEnv):
    def __init__(self, **kwargs):
        self.env_args = kwargs["env_args"]
        self.deliveryed_visible = self.env_args["deliveryed_visible"]
        self.min_direct_dist = self.env_args['min_direct_dist']
        self.info_batch_size = self.batch_size if self.env_args["info_batch_size"] == -1 else self.env_args["info_batch_size"]
        self.debug = self.env_args["debug"] # all envs use one seed.
        self.algorithm = kwargs["algorithm"]

        self.place = kwargs["place"]
        self.is_train = kwargs["is_train"]
        self.suffix = kwargs["suffix"]
        self.pack_size = self.env_args["pack_size"]
        
        self.drone_speed_radio = 3
        self.drone_add_radio = 3
        
        self.drone_cost_ratio = self.env_args['dist_cost_drone']
        self.courier_cost_ratio = self.env_args['dist_cost_courier']
        self.request_profit_ratio = self.env_args['dist_req_profit']
        # max_value: 规定时间内获得最高的价值
        # min_dist: 完成所有任务的路程最少，可能有完不成的风险。可能造成智能体不动的情况，如何通过加 mask 解决。
        self.config = {
            "tw": {
                "n_node": 36,
                "n_courier": 20,
                "n_drone": 20,
                "n_station": 5,
                "n_tot_requests": 400,
                "except": [],
                "pack_size": 5,
                'stations': ["wsqq3", "wsqqh", "wsqqe", "wsqqw", "wsqqv"]
            },
            "sg": {
                "n_node": 36,
                "n_courier": 20,
                "n_drone": 20,
                "n_station": 5,
                "n_tot_requests": 300,
                "except": [],
                "pack_size": 5,
                'stations': ["w21zd", "w21zk", "w21zv", "w23b1", "w21xx"]
            },
            "se": {
                "n_node": 36,
                "n_courier": 20,
                "n_drone": 20,
                "n_station": 5,
                "n_tot_requests": 120,
                "pack_size": 3,
                'stations': ["u6sc0", "u6scc", "u6sce", "u6scq", "u6sch"],
                "except": ['u7xqm']
            },
            
        }[self.place]
        print(self.place)
        self.pack_size = self.config["pack_size"]
        
        self.n_node = self.config["n_node"]
        self.n_tot_requests = self.config["n_tot_requests"]
        self.n_courier = self.config["n_courier"]
        self.n_drone = self.config['n_drone']
        self.n_station = self.config["n_station"]

        # Dataset specific parameter
        self.env_args["n_node"] = self.n_node
        self.env_args["n_courier"] = self.n_courier
        self.env_args['n_drone'] = self.n_drone
        self.env_args["n_tot_requests"] = self.n_tot_requests
        if "max_consider_requests" in self.config:
            self.env_args["max_consider_requests"] = self.config["max_consider_requests"]
        self.max_consider_requests = self.n_tot_requests if self.env_args["max_consider_requests"] == -1 else self.env_args["max_consider_requests"]

        
        self.device = kwargs["device"]
        self.rng = torch.Generator(self.device)
        self.rng_env = torch.Generator(self.device)
        self.genearte()

        if self.is_train:
            self.batch_size = kwargs["sample_batch_size"]
        else:
            self.batch_size = min(self.cached["requests_from"].shape[0], kwargs["sample_batch_size"])
        self.batch_arange = torch.arange(self.batch_size, device=self.device)
        self.node_arange = torch.arange(self.n_node, device=self.device)
        self.requests_arange = torch.arange(self.n_tot_requests, device=self.device)
        

    def genearte(self):
        delta = 0.02197265625
        interval = 30  # minutes
        n_node = self.n_node
        n_station = self.n_station
        except_node = self.config["except"]
        min_direct_dist = getattr(self, 'min_direct_dist', 1) 
        assert n_node == self.env_args["n_node"]
        
        # 读取原始数据
        orders = pd.read_csv(f"data/data_{self.place}/orders_{self.place}_{self.suffix}.txt", index_col=0)
        vendors = pd.read_csv(f"data/data_{self.place}/vendors_{self.place}.txt", index_col=0)
        
        # ===== 第一步：计算所有订单的距离并筛选长距离订单 =====
        print("步骤1: 筛选长距离订单...")
        
        # 为所有订单添加坐标信息
        orders_with_coords = orders.copy()
        orders_with_coords['customer_lat'] = orders_with_coords['geohash'].apply(
            lambda x: geohash.decode(x)[0] if pd.notna(x) else None
        )
        orders_with_coords['customer_lon'] = orders_with_coords['geohash'].apply(
            lambda x: geohash.decode(x)[1] if pd.notna(x) else None
        )
        
        # 加入vendor坐标
        orders_with_coords = orders_with_coords.join(
            vendors[["vendor_id", "geohash"]].set_index("vendor_id"), 
            on="vendor_id", 
            how="left", 
            rsuffix="_v"
        )
        orders_with_coords['vendor_lat'] = orders_with_coords['geohash_v'].apply(
            lambda x: geohash.decode(x)[0] if pd.notna(x) else None
        )
        orders_with_coords['vendor_lon'] = orders_with_coords['geohash_v'].apply(
            lambda x: geohash.decode(x)[1] if pd.notna(x) else None
        )
        
        # 计算距离（以delta为单位）
        orders_with_coords['distance'] = (
            np.abs(orders_with_coords['customer_lat'] - orders_with_coords['vendor_lat']) + 
            np.abs(orders_with_coords['customer_lon'] - orders_with_coords['vendor_lon'])
        ) / delta
        orders_with_coords['distance'] = (orders_with_coords['distance'] + 0.5).astype(int)
        orders_with_coords['distance'] = orders_with_coords['distance'] // 2
        
        # 筛选长距离订单
        long_distance_orders = orders_with_coords[
            orders_with_coords['distance'] >= min_direct_dist
        ].copy()
        
        print(f"原始订单数: {len(orders)}")
        print(f"长距离订单数 (>={min_direct_dist}): {len(long_distance_orders)}")
        print(f"距离统计: min={orders_with_coords['distance'].min():.2f}, "
            f"max={orders_with_coords['distance'].max():.2f}, "
            f"mean={orders_with_coords['distance'].mean():.2f}")
        
        # ===== 第二步：基于长距离订单选择高频节点 =====
        print("\n步骤2: 基于长距离订单选择节点...")
        
        # 统计长距离订单中的高频vendor
        vendors_geohash_counts = long_distance_orders['geohash_v'].value_counts()
        print(f"长距离订单涉及的vendor数量: {len(vendors_geohash_counts)}")
        
        # 统计长距离订单中的高频customer节点
        customer_geohash_counts = long_distance_orders['geohash'].value_counts()
        
        # 排除特定节点
        customer_geohash_counts = customer_geohash_counts.drop(
            except_node, errors='ignore'
        ).drop(
            vendors_geohash_counts.index, errors='ignore'
        )
        
        # 选择节点：优先选择高频vendor，然后补充高频customer节点
        # n_vendor_nodes = min(len(vendors_geohash_counts), n_node // 3)  # vendor节点不超过总数的1/3
        n_vendor_nodes = len(vendors_geohash_counts)
        selected_vendors = vendors_geohash_counts.head(n_vendor_nodes).index.tolist()
        
        n_customer_nodes = n_node - len(selected_vendors)
        selected_customers = customer_geohash_counts.head(n_customer_nodes).index.tolist()
        nodes = selected_customers + selected_vendors
        print(f"选择的vendor节点数: {len(selected_vendors)}")
        print(f"选择的customer节点数: {len(selected_customers)}")
        assert len(nodes) == n_node
        nodes.sort()
        
        # 验证坐标精度
        nodes_coords = np.array([geohash.decode(code) for code in nodes])
        for code in nodes:
            _, _, dlat, dlon = geohash.decode(code, True)
            assert dlat == delta and dlon == delta
        
        # ===== 第三步：过滤订单并计算节点距离 =====
        print("\n步骤3: 处理订单数据...")
        
        # 重新处理orders，只保留节点在选定范围内的订单
        orders = orders.join(
            vendors[["vendor_id", "geohash"]].set_index("vendor_id"), 
            on="vendor_id", 
            how="left", 
            rsuffix="_v"
        )
        
        # 筛选：vendor和customer都在选定节点中
        orders = orders[orders.geohash.isin(nodes) & orders.geohash_v.isin(nodes)]
        
        # 排除同一节点的订单
        orders = orders[orders.geohash != orders.geohash_v]
        orders = orders.reset_index(drop=True)
        
        # 添加节点索引
        node_to_idx = {code: idx for idx, code in enumerate(nodes)}
        orders["from"] = orders["geohash_v"].map(node_to_idx)
        orders["to"] = orders["geohash"].map(node_to_idx)
        
        # 计算节点间距离矩阵
        node_node = (np.abs(nodes_coords[:, None] - nodes_coords[None]).sum(-1) / delta + 0.5).astype(int)
        assert (node_node % 2 == 0).all()
        node_node = node_node // 2
        self.node_node = node_node
        # assert node_node.max() <= self.env_args["max_dist"]
        
        # 再次筛选：只保留距离 >= min_direct_dist 的订单
        orders['distance'] = orders.apply(lambda row: node_node[row['from'], row['to']], axis=1)
        orders = orders[orders['distance'] >= min_direct_dist]
        
        print(f"最终保留订单数: {len(orders)}")
        print(f"订单距离统计: min={orders['distance'].min()}, "
            f"max={orders['distance'].max()}, "
            f"mean={orders['distance'].mean():.2f}")
        
        # ===== 第四步：选择无人机站点 =====
        print("\n步骤4: 选择无人机站点...")
        if 'stations' in self.config and self.config['stations'] is not None:
            # 使用配置的站点
            station_geohash = self.config['stations']
            node_to_idx = {code: idx for idx, code in enumerate(nodes)}
            station_idx = [node_to_idx[h] for h in station_geohash if h in node_to_idx]
            station_idx = np.array(station_idx, dtype=np.int64)
        else:
            # 随机选择站点
            n_station = self.n_station
            station_idx = np.sort(np.random.choice(n_node, n_station, replace=False))

        station_idx = np.concatenate([station_idx, [n_node]])  # 最后一位为 n_node
        station_mask = np.zeros(n_node, dtype=bool)
        station_mask[station_idx[:-1]] = True
        station_node_node = node_node[np.ix_(station_idx[:-1], station_idx[:-1])]
        
        self.station_idx = station_idx
        self.station_mask = station_mask
        self.station_node_node = station_node_node
        
        # 过滤掉 from/to 为 station 的订单
        orders = orders[~orders["from"].isin(station_idx[:-1])]
        orders = orders[~orders["to"].isin(station_idx[:-1])]
        
        print(f"过滤站点订单后剩余: {len(orders)}")
        
        # ===== 第五步：为订单分配最近站点 =====
        print("\n步骤5: 分配最近站点...")
        
        orders["order_time"] = pd.to_datetime(orders["order_time"], format="%H:%M:%S")
        
        node_node_tensor = torch.tensor(node_node, dtype=torch.int64)
        from_nodes = torch.tensor(orders["from"].values)
        to_nodes = torch.tensor(orders["to"].values)
        station_nodes = torch.tensor(station_idx[:-1])
        
        # 最近的 station1（起点站）
        dist_from = node_node_tensor[from_nodes][:, station_nodes]
        nearest_station1 = station_nodes[dist_from.argmin(dim=1)]
        
        # 最近的 station2（终点站）
        dist_to = node_node_tensor[to_nodes][:, station_nodes]
        nearest_station2 = station_nodes[dist_to.argmin(dim=1)]
        
        orders["station1"] = nearest_station1.cpu().numpy()
        orders["station2"] = nearest_station2.cpu().numpy()
        
        # ===== 第六步：生成时间帧和请求批次 =====
        print("\n步骤6: 生成请求批次...")
        
        assert 24 * 60 % interval == 0
        n_norequest_frame = 1
        n_requested_frame = 24 * 60 // interval
        n_frame = n_requested_frame + n_norequest_frame
        self.n_frame = n_frame
        assert self.n_frame == self.env_args["n_frame"]
        
        groups = list(orders.groupby(orders["order_day"]))
        n_inst = len(groups)
        
        requests_from_batched = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        requests_to_batched = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        requests_station1_batched = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        requests_station2_batched = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        requests_value_batched = torch.zeros(n_inst, self.n_tot_requests, dtype=torch.int64, device=self.device)
        frame_n_accumulate_requests_batched = []
        
        for inst_idx, (_, day_df) in enumerate(groups):
            requests_from = []
            requests_to = []
            requests_value = []
            requests_occur = []
            requests_station1 = []
            requests_station2 = []
            
            for (_from, _to), uv_df in day_df.groupby(["from", "to"]):
                # 确保订单满足距离阈值（前面已经过滤过，这里是双保险）
                # if node_node[_from, _to] < min_direct_dist:
                #     continue
                
                uv_df = uv_df.sort_values("order_time")
                minutes = uv_df["order_time"].dt.hour * 60 + uv_df["order_time"].dt.minute
                last = -1
                
                for i in range(len(uv_df)):
                    if i - last >= self.pack_size:
                        requests_from.append(_from)
                        requests_to.append(_to)
                        requests_value.append(i - last)
                        requests_occur.append(minutes.iloc[i] // interval)
                        requests_station1.append(uv_df["station1"].iloc[i])
                        requests_station2.append(uv_df["station2"].iloc[i])
                        last = i
            
            # 按时间排序
            requests_occur = torch.tensor(requests_occur)[:self.n_tot_requests]
            reindex = requests_occur.argsort()
            
            requests_from = torch.tensor(requests_from)[:self.n_tot_requests][reindex]
            requests_to = torch.tensor(requests_to)[:self.n_tot_requests][reindex]
            requests_value = torch.tensor(requests_value)[:self.n_tot_requests][reindex]
            requests_station1 = torch.tensor(requests_station1)[:self.n_tot_requests][reindex]
            requests_station2 = torch.tensor(requests_station2)[:self.n_tot_requests][reindex]
            requests_occur = requests_occur[reindex]
            
            frame_n_accumulate_requests_batched.append(
                torch.bincount(requests_occur, minlength=self.n_frame + 1)
            )
            
            n_req = requests_from.shape[0]
            requests_from_batched[inst_idx, :n_req] = requests_from.to(self.device)
            requests_to_batched[inst_idx, :n_req] = requests_to.to(self.device)
            requests_value_batched[inst_idx, :n_req] = requests_value.to(self.device)
            requests_station1_batched[inst_idx, :n_req] = requests_station1.to(self.device)
            requests_station2_batched[inst_idx, :n_req] = requests_station2.to(self.device)
        
        frame_n_accumulate_requests_batched = torch.stack(frame_n_accumulate_requests_batched).to(self.device)
        frame_n_accumulate_requests_batched = frame_n_accumulate_requests_batched.cumsum(-1)
        
        print(f"\n每日请求数: {frame_n_accumulate_requests_batched[:, -1].cpu().tolist()}")
        print("生成实例完成。")
        
        # ===== 计算最终配送距离统计 =====
        # node_node_tensor = torch.tensor(self.node_node, dtype=torch.float32, device=self.device)
        # valid_mask = (requests_from_batched > 0) & (requests_to_batched > 0)
        # from_all = requests_from_batched[valid_mask]
        # to_all = requests_to_batched[valid_mask]
        # distances = node_node_tensor[from_all, to_all]
        
        # avg_distance = distances.mean().item()
        # min_distance = distances.min().item()
        # max_distance = distances.max().item()
        
        # print(f"\n最终配送距离统计:")
        # print(f"  最小: {min_distance:.2f}")
        # print(f"  最大: {max_distance:.2f}")
        # print(f"  平均: {avg_distance:.2f}")
        # # ===== 可视化节点分布 =====
        # latitudes = nodes_coords[:, 0]
        # longitudes = nodes_coords[:, 1]
        
        # plt.figure(figsize=(10, 8))
        # plt.scatter(longitudes, latitudes, c='blue', s=30, label='Customer Nodes', alpha=0.6)
        
        # # 标记vendor节点
        # vendor_coords = np.array([geohash.decode(code) for code in selected_vendors])
        # plt.scatter(vendor_coords[:, 1], vendor_coords[:, 0], 
        #             c='red', s=80, label='Vendors', alpha=0.9, marker='x', linewidths=2)
        
        # # 标记station节点
        # plt.scatter(nodes_coords[station_idx[:-1], 1], nodes_coords[station_idx[:-1], 0], 
        #             c='green', s=100, label='Drone Stations', marker='s', alpha=0.9)
        
        # plt.title(f"Node Distribution - {self.place}\n"
        #         f"(min_dist>={min_direct_dist}, avg_dist={avg_distance:.2f})")
        # plt.xlabel("Longitude")
        # plt.ylabel("Latitude")
        # plt.legend(loc='best')
        # plt.grid(True, linestyle='--', alpha=0.3)
        # plt.tight_layout()
        # plt.savefig("node_distribution.png", dpi=300, bbox_inches='tight')
        # plt.close()
        
        # print("\n选择的无人机站点 geohash：")
        # for i, idx in enumerate(station_idx[:-1]):  # 排除最后一个哨兵 n_node
        #     print(f"  Station {i}: {nodes[idx]}")
        
        # print("节点分布图已保存: node_distribution.png")
        # ===== 保存到缓存 =====
        self.cached = {}
        self.cached["coord"] = torch.from_numpy(nodes_coords).to(self.device)
        self.cached["requests_from"] = requests_from_batched
        self.cached["requests_to"] = requests_to_batched
        self.cached["requests_value"] = requests_value_batched
        self.cached["requests_volumn"] = torch.ones_like(requests_value_batched)
        self.cached["requests_station1"] = requests_station1_batched
        self.cached["requests_station2"] = requests_station2_batched
        self.cached["node_node"] = torch.from_numpy(node_node).to(self.device)
        
        self.cached["station_idx"] = torch.from_numpy(station_idx).to(self.device)
        self.cached["station_mask"] = torch.from_numpy(station_mask).to(self.device)
        self.cached["station_node_node"] = torch.from_numpy(station_node_node).to(self.device)
        
        self.cached["frame_n_accumulate_requests"] = frame_n_accumulate_requests_batched
        # assert(False)
        
    def reset(self, **kwargs):    
        station_idx = self.cached["station_idx"].unsqueeze(0).repeat(self.batch_size, 1)
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

        if not self.debug:
            cycle_indices = torch.arange(self.n_drone, device=self.device) % self.n_station
            # 为每个batch添加随机偏移，避免所有batch都有相同的分布模式
            batch_offsets = torch.randint(0, self.n_station, (self.batch_size,), 
                                        generator=self.rng, device=self.device)
            
            # 应用偏移并取模
            rand_idx = (cycle_indices.unsqueeze(0) + batch_offsets.unsqueeze(1)) % self.n_station
            
            drones_target = torch.gather(
                station_idx[:,:-1].unsqueeze(1).expand(-1, self.n_drone, -1),
                2,
                rand_idx.unsqueeze(2)
            ).squeeze(2)
        else:
            # debug 模式下，先生成 [1, n_drone] 的 rand_idx，再 repeat 到整个 batch
            rand_idx = torch.randint(
                0, self.n_station,
                (1, self.n_drone),
                generator=self.rng,
                device=self.device
            )
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
        visible[self.requests_arange < self.cached["frame_n_accumulate_requests"][batch_idx, :1]] = True
        self.requests = {
            "from": self.cached["requests_from"][batch_idx],
            "to": self.cached["requests_to"][batch_idx],
            "value": self.cached["requests_value"][batch_idx],
            "volumn": self.cached["requests_volumn"][batch_idx],
            "visible": visible,
            "appear": (self.requests_arange[None, :, None] < self.cached["frame_n_accumulate_requests"][batch_idx, None, :]).to(torch.int64).argmax(-1),
            "station1": torch.full([self.batch_size, self.n_tot_requests], self.n_node, device=self.device),
            "station2": torch.full([self.batch_size, self.n_tot_requests], self.n_node, device=self.device),
        }
        rel_mat_courier = torch.full([self.batch_size, self.n_courier, self.n_tot_requests], RELATION.no_relation.value, device=self.device)
        rel_mat_courier.masked_fill_((self.requests_arange >= self.cached["frame_n_accumulate_requests"][batch_idx, -1:])[:, None, :], RELATION.padded.value)
        
        rel_mat_drone = torch.full([self.batch_size, self.n_drone, self.n_tot_requests], RELATION.no_relation.value, device=self.device)
        rel_mat_drone.masked_fill_((self.requests_arange >= self.cached["frame_n_accumulate_requests"][batch_idx, -1:])[:, None, :], RELATION.padded.value)
        
        self._global = {
            "frame": 0,
            "n_exist_requests": self.cached["frame_n_accumulate_requests"][batch_idx, 0],
            "node_node": self.cached["node_node"].repeat(self.batch_size, 1, 1), 
            "station_idx": self.cached["station_idx"].unsqueeze(0).repeat(self.batch_size, 1),
            "station_mask": self.cached["station_mask"].unsqueeze(0).repeat(self.batch_size, 1),
            "station_node_node": self.cached["station_node_node"].repeat(self.batch_size, 1, 1), 
            "courier_request" : rel_mat_courier,
            "drone_request" : rel_mat_drone,
        }
        
        self.node_node = self._global['node_node']
        self.station_node_node = self._global['station_node_node']
        self.station_mask = self._global['station_mask']
        self.station_idx = self._global['station_idx']

        self.frame_n_accumulate_requests = self.cached["frame_n_accumulate_requests"][batch_idx]

        # import ipdb;ipdb.set_trace()
        # theoretical server rate
        global occupancy_logged
        if not occupancy_logged:
            _all_workload = self._global["node_node"][self.batch_arange[:, None], self.requests["from"], self.requests["to"]].float().sum(-1).mean()
            _max_capacity = self.n_courier * self.env_args["n_frame"]# * self.env_args["max_capacity"]
            print("theoretical server rate:", (_max_capacity / _all_workload).item())
            occupancy_logged = True

        self.pre_obs, self.pre_unobs = self.get_obs()
        return self.pre_obs