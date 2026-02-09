#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 DroneTransferEnvLADE 环境的正确性
- 查看选取的节点
- 在地图上绘制节点
- 统计每天生成的订单数
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import ndimage
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.mvdpdp.env_lade import DroneTransferEnvLADE, CITY_CONFIGS

def create_test_env_args():
    """创建测试用的环境参数"""
    env_args = {
        "scenario": "lade",
        "interruptable": False,
        "use_sp": True,
        "n_frame": 80,
        "n_requested_frame": 60,
        "n_node": 30,
        "n_courier": 10,
        "n_station": 5,
        "n_drone": 5,
        "min_direct_dist": 5,  # 最小直接距离（km）
        "gen_mode": "direct",
        "n_init_requests": 20,
        "n_norm_requests": 100,
        "max_consider_requests": -1,
        "max_capacity": 5,
        "dist_distribution": "2D",
        "max_dist": 10,
        "n_node_tot": 100,
        "dist_cost": 0.5,
        "dist_cost_drone": 0.3,
        "dist_cost_courier": 0.2,
        "dist_req_profit": 1,
        "info_batch_size": -1,
        "debug": False,
        "deliveryed_visible": True,
    }
    return env_args

def plot_nodes_on_map(env, city="hz", save_path="nodes_map.png"):
    """在地图上绘制节点"""
    # 获取节点坐标
    node_coords = env.cached["coord"].cpu().numpy()  # (n_node, 2) where 2 is (lng, lat)
    station_indices = env.cached["station_idx"].cpu().numpy()  # (n_station+1,)
    station_mask = env.cached["station_mask"].cpu().numpy()  # (n_node,)
    
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
        linewidth=2, edgecolor='gray', facecolor='lightgray', alpha=0.3, label='城市边界'
    )
    ax.add_patch(rect)
    
    # 绘制所有节点
    pickup_nodes = []
    delivery_nodes = []
    station_nodes = []
    
    n_pickup = int(env.n_node * env.pickup_node_ratio)
    
    for i in range(env.n_node):
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
                  alpha=0.7, label=f'Pickup节点 ({len(pickup_nodes)})', edgecolors='darkblue', linewidths=1.5)
    
    # 绘制delivery节点
    if delivery_nodes:
        delivery_lngs, delivery_lats = zip(*delivery_nodes)
        ax.scatter(delivery_lngs, delivery_lats, c='green', marker='s', s=100, 
                  alpha=0.7, label=f'Delivery节点 ({len(delivery_nodes)})', edgecolors='darkgreen', linewidths=1.5)
    
    # 绘制站点
    if station_nodes:
        station_lngs, station_lats = zip(*station_nodes)
        ax.scatter(station_lngs, station_lats, c='red', marker='^', s=200, 
                  alpha=0.9, label=f'站点 ({len(station_nodes)})', edgecolors='darkred', linewidths=2)
    
    # 设置坐标轴
    ax.set_xlabel('经度 (Longitude)', fontsize=12)
    ax.set_ylabel('纬度 (Latitude)', fontsize=12)
    ax.set_title(f'{city.upper()} 城市节点分布图', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # 设置坐标轴范围（稍微扩展一点以便查看）
    lng_margin = (lng_range[1] - lng_range[0]) * 0.05
    lat_margin = (lat_range[1] - lat_range[0]) * 0.05
    ax.set_xlim(lng_range[0] - lng_margin, lng_range[1] + lng_margin)
    ax.set_ylim(lat_range[0] - lat_margin, lat_range[1] + lat_margin)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"节点地图已保存到: {save_path}")
    plt.close()

def plot_heatmap_with_nodes(env, city="hz", save_path="nodes_heatmap.png", is_train=True):
    """在地图上绘制节点和订单热力图"""
    # 获取节点坐标
    node_coords = env.cached["coord"].cpu().numpy()  # (n_node, 2) where 2 is (lng, lat)
    station_indices = env.cached["station_idx"].cpu().numpy()  # (n_station+1,)
    station_mask = env.cached["station_mask"].cpu().numpy()  # (n_node,)
    
    # 获取订单数据
    if is_train:
        requests_from = env.cached["train_requests_from"]
        requests_to = env.cached["train_requests_to"]
        dataset_name = "训练集"
    else:
        requests_from = env.cached["test_requests_from"]
        requests_to = env.cached["test_requests_to"]
        dataset_name = "测试集"
    
    # 获取城市配置
    config = CITY_CONFIGS[city]
    lng_range = config["lng_range"]
    lat_range = config["lat_range"]
    
    # 收集所有订单的起点和终点坐标
    pickup_coords = []
    delivery_coords = []
    all_coords = []
    
    n_instances = requests_from.shape[0]
    for i in range(n_instances):
        valid_mask = (requests_from[i] < env.n_node) & (requests_to[i] < env.n_node)
        valid_from = requests_from[i][valid_mask].cpu().numpy()
        valid_to = requests_to[i][valid_mask].cpu().numpy()
        
        for f, t in zip(valid_from, valid_to):
            if f < env.n_node and t < env.n_node:
                pickup_coords.append(node_coords[f])
                delivery_coords.append(node_coords[t])
                all_coords.append(node_coords[f])
                all_coords.append(node_coords[t])
    
    if len(all_coords) == 0:
        print(f"警告: {dataset_name}没有有效订单，跳过热力图绘制")
        return
    
    # 转换为numpy数组
    pickup_coords = np.array(pickup_coords)
    delivery_coords = np.array(delivery_coords)
    all_coords = np.array(all_coords)
    
    # 创建图形，包含三个子图
    fig = plt.figure(figsize=(18, 10))
    
    # 子图1: Pickup热力图
    ax1 = plt.subplot(1, 3, 1)
    plot_single_heatmap(ax1, pickup_coords, node_coords, station_mask, 
                       lng_range, lat_range, city, "Pickup订单热力图", 'Blues')
    
    # 子图2: Delivery热力图
    ax2 = plt.subplot(1, 3, 2)
    plot_single_heatmap(ax2, delivery_coords, node_coords, station_mask, 
                       lng_range, lat_range, city, "Delivery订单热力图", 'Reds')
    
    # 子图3: 合并热力图
    ax3 = plt.subplot(1, 3, 3)
    plot_single_heatmap(ax3, all_coords, node_coords, station_mask, 
                       lng_range, lat_range, city, "全部订单热力图", 'YlOrRd')
    
    plt.suptitle(f'{city.upper()} 城市订单热力图 ({dataset_name})', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"订单热力图已保存到: {save_path}")
    print(f"  Pickup订单数: {len(pickup_coords)}")
    print(f"  Delivery订单数: {len(delivery_coords)}")
    print(f"  总订单坐标数: {len(all_coords)}")
    plt.close()

def plot_single_heatmap(ax, coords, node_coords, station_mask, lng_range, lat_range, 
                       city, title, colormap='YlOrRd'):
    """绘制单个热力图子图"""
    # 绘制城市边界框
    rect = plt.Rectangle(
        (lng_range[0], lat_range[0]),
        lng_range[1] - lng_range[0],
        lat_range[1] - lat_range[0],
        linewidth=1.5, edgecolor='gray', facecolor='lightgray', alpha=0.2
    )
    ax.add_patch(rect)
    
    # 计算热力图
    n_bins = 50
    lng_bins = np.linspace(lng_range[0], lng_range[1], n_bins + 1)
    lat_bins = np.linspace(lat_range[0], lat_range[1], n_bins + 1)
    
    # 计算2D直方图
    heatmap, xedges, yedges = np.histogram2d(
        coords[:, 0], coords[:, 1], 
        bins=[lng_bins, lat_bins]
    )
    
    # 对热力图进行平滑处理
    heatmap = ndimage.gaussian_filter(heatmap, sigma=1.0)
    
    # 转置以便正确显示（histogram2d返回的是(lat, lng)顺序）
    heatmap = heatmap.T
    
    # 计算网格中心
    lng_centers = (xedges[:-1] + xedges[1:]) / 2
    lat_centers = (yedges[:-1] + yedges[1:]) / 2
    
    # 绘制热力图
    extent = [lng_range[0], lng_range[1], lat_range[0], lat_range[1]]
    im = ax.imshow(heatmap, extent=extent, origin='lower', 
                   cmap=colormap, alpha=0.6, aspect='auto', interpolation='bilinear')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('订单密度', fontsize=10)
    
    # 绘制节点
    n_pickup = int(len(node_coords) * 0.4)
    pickup_nodes = []
    delivery_nodes = []
    station_nodes = []
    
    for i in range(len(node_coords)):
        lng, lat = node_coords[i]
        if station_mask[i]:
            station_nodes.append((lng, lat))
        elif i < n_pickup:
            pickup_nodes.append((lng, lat))
        else:
            delivery_nodes.append((lng, lat))
    
    # 绘制节点（使用较小的标记以便看清热力图）
    if pickup_nodes:
        pickup_lngs, pickup_lats = zip(*pickup_nodes)
        ax.scatter(pickup_lngs, pickup_lats, c='blue', marker='o', s=30, 
                  alpha=0.6, edgecolors='darkblue', linewidths=0.5)
    
    if delivery_nodes:
        delivery_lngs, delivery_lats = zip(*delivery_nodes)
        ax.scatter(delivery_lngs, delivery_lats, c='green', marker='s', s=30, 
                  alpha=0.6, edgecolors='darkgreen', linewidths=0.5)
    
    if station_nodes:
        station_lngs, station_lats = zip(*station_nodes)
        ax.scatter(station_lngs, station_lats, c='red', marker='^', s=80, 
                  alpha=0.8, edgecolors='darkred', linewidths=1)
    
    # 设置坐标轴
    ax.set_xlabel('经度 (Longitude)', fontsize=10)
    ax.set_ylabel('纬度 (Latitude)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # 设置坐标轴范围
    lng_margin = (lng_range[1] - lng_range[0]) * 0.05
    lat_margin = (lat_range[1] - lat_range[0]) * 0.05
    ax.set_xlim(lng_range[0] - lng_margin, lng_range[1] + lng_margin)
    ax.set_ylim(lat_range[0] - lat_margin, lat_range[1] + lat_margin)

def analyze_orders(env, is_train=True):
    """分析订单数据"""
    if is_train:
        requests_from = env.cached["train_requests_from"]
        requests_to = env.cached["train_requests_to"]
        frame_accumulate = env.cached["train_frame_accumulate"]
        dataset_name = "训练集"
    else:
        requests_from = env.cached["test_requests_from"]
        requests_to = env.cached["test_requests_to"]
        frame_accumulate = env.cached["test_frame_accumulate"]
        dataset_name = "测试集"
    
    n_instances = requests_from.shape[0]
    n_frame = env.env_args["n_frame"]
    
    print(f"\n{'='*60}")
    print(f"{dataset_name}订单分析")
    print(f"{'='*60}")
    print(f"实例数量: {n_instances}")
    print(f"总时间步: {n_frame}")
    print(f"每实例订单数: {env.n_tot_requests}")
    
    # 统计每天的订单数
    print(f"\n各实例的订单统计:")
    print(f"{'实例ID':<10} {'总订单数':<12} {'初始订单':<12} {'后续订单':<12} {'平均每步订单':<15}")
    print("-" * 70)
    
    total_orders_all = 0
    for i in range(n_instances):
        # 计算有效订单数（非padding）
        valid_mask = (requests_from[i] < env.n_node) & (requests_to[i] < env.n_node)
        n_valid = valid_mask.sum().item()
        
        # 初始订单数（时间步0）
        n_init = frame_accumulate[i, 0].item()
        
        # 后续订单数
        n_later = n_valid - n_init
        
        # 平均每步订单数
        avg_per_frame = n_valid / n_frame if n_frame > 0 else 0
        
        total_orders_all += n_valid
        
        print(f"{i:<10} {n_valid:<12} {n_init:<12} {n_later:<12} {avg_per_frame:<15.2f}")
    
    print("-" * 70)
    print(f"平均每实例订单数: {total_orders_all / n_instances:.2f}")
    
    # 输出每一步新出现的订单数
    print(f"\n各实例每一步新出现的订单数 (显示前3个实例):")
    for i in range(min(3, n_instances)):  # 只显示前3个实例，避免输出过多
        print(f"\n实例 {i}:")
        print(f"{'时间步':<10} {'累积订单数':<15} {'新出现订单数':<15}")
        print("-" * 45)
        
        # 计算每一步新出现的订单数
        frame_accumulate_np = frame_accumulate[i].cpu().numpy()
        
        # 第0步新出现的订单数
        new_orders_step0 = frame_accumulate_np[0]
        print(f"{0:<10} {frame_accumulate_np[0]:<15} {new_orders_step0:<15}")
        
        # 后续步骤新出现的订单数
        shown_count = 0
        for t in range(1, min(n_frame + 1, len(frame_accumulate_np))):
            new_orders = frame_accumulate_np[t] - frame_accumulate_np[t-1]
            # 显示前15步或新订单数大于0的步骤
            if new_orders > 0 or t <= 15:
                print(f"{t:<10} {frame_accumulate_np[t]:<15} {new_orders:<15}")
                shown_count += 1
            elif shown_count > 0:
                # 如果已经显示了一些步骤，但当前步骤没有新订单，跳过
                continue
        
        # 检查是否还有未显示的步骤
        last_shown = min(15, len(frame_accumulate_np) - 1)
        if len(frame_accumulate_np) > last_shown + 1:
            remaining_new = sum(frame_accumulate_np[t] - frame_accumulate_np[t-1] 
                              for t in range(last_shown + 1, len(frame_accumulate_np)))
            if remaining_new > 0:
                print(f"  ... (还有 {len(frame_accumulate_np) - last_shown - 1} 个时间步，其中 {remaining_new} 个新订单)")
    
    # 统计所有实例的平均每步新订单数
    print(f"\n所有实例平均每步新出现的订单数:")
    print(f"{'时间步':<10} {'平均新订单数':<15} {'最大新订单数':<15} {'最小新订单数':<15}")
    print("-" * 60)
    
    # 计算所有实例的平均值
    frame_accumulate_all = frame_accumulate.cpu().numpy()  # (n_instances, n_frame+1)
    
    # 第0步
    new_orders_step0_all = frame_accumulate_all[:, 0]
    print(f"{0:<10} {new_orders_step0_all.mean():<15.2f} {new_orders_step0_all.max():<15} {new_orders_step0_all.min():<15}")
    
    # 后续步骤
    for t in range(1, min(n_frame + 1, frame_accumulate_all.shape[1])):
        new_orders_all = frame_accumulate_all[:, t] - frame_accumulate_all[:, t-1]
        if new_orders_all.sum() > 0 or t <= 10:  # 显示前10步或仍有新订单的步骤
            print(f"{t:<10} {new_orders_all.mean():<15.2f} {new_orders_all.max():<15} {new_orders_all.min():<15}")
        elif t == 11:
            print("  ... (后续步骤无新订单)")
            break
    
    # 统计订单距离分布
    node_node = env.cached["node_node"].cpu().numpy()
    distances = []
    for i in range(n_instances):
        valid_mask = (requests_from[i] < env.n_node) & (requests_to[i] < env.n_node)
        valid_from = requests_from[i][valid_mask].cpu().numpy()
        valid_to = requests_to[i][valid_mask].cpu().numpy()
        
        for f, t in zip(valid_from, valid_to):
            if f < env.n_node and t < env.n_node:
                distances.append(node_node[f, t])
    
    distances_array = None
    if len(distances) > 0:
        distances_array = np.array(distances)
        print(f"\n订单距离统计:")
        print(f"  最小距离: {distances_array.min()} km")
        print(f"  最大距离: {distances_array.max()} km")
        print(f"  平均距离: {distances_array.mean():.2f} km")
        print(f"  中位数距离: {np.median(distances_array):.2f} km")
        print(f"  标准差: {distances_array.std():.2f} km")
    
    return {
        'n_instances': n_instances,
        'total_orders': total_orders_all,
        'avg_orders_per_instance': total_orders_all / n_instances,
        'distances': distances_array
    }

def print_node_info(env):
    """打印节点信息"""
    node_coords = env.cached["coord"].cpu().numpy()
    station_indices = env.cached["station_idx"].cpu().numpy()
    station_mask = env.cached["station_mask"].cpu().numpy()
    
    n_pickup = int(env.n_node * env.pickup_node_ratio)
    
    print(f"\n{'='*60}")
    print("节点信息")
    print(f"{'='*60}")
    print(f"总节点数: {env.n_node}")
    print(f"Pickup节点数: {n_pickup} ({env.pickup_node_ratio*100:.0f}%)")
    print(f"Delivery节点数: {env.n_node - n_pickup} ({(1-env.pickup_node_ratio)*100:.0f}%)")
    print(f"站点数: {env.n_station}")
    print(f"\n节点坐标范围:")
    print(f"  经度: [{node_coords[:, 0].min():.6f}, {node_coords[:, 0].max():.6f}]")
    print(f"  纬度: [{node_coords[:, 1].min():.6f}, {node_coords[:, 1].max():.6f}]")
    
    print(f"\n站点索引: {station_indices[:-1].tolist()}")
    print(f"\n前10个节点坐标 (lng, lat):")
    for i in range(min(10, env.n_node)):
        node_type = "站点" if station_mask[i] else ("Pickup" if i < n_pickup else "Delivery")
        print(f"  节点 {i:2d} ({node_type:8s}): ({node_coords[i, 0]:.6f}, {node_coords[i, 1]:.6f})")

def main():
    """主测试函数"""
    print("开始测试 DroneTransferEnvLADE 环境...")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境参数
    env_args = create_test_env_args()
    
    # 测试不同城市
    cities = ["hz", "cq", "sh"]
    
    for city in cities:
        print(f"\n{'#'*60}")
        print(f"测试城市: {city.upper()}")
        print(f"{'#'*60}")
        
        try:
            # 创建训练环境
            print("\n创建训练环境...")
            env_train = DroneTransferEnvLADE(
                env_args=env_args,
                algorithm="BR",
                city=city,
                is_train=True,
                device=device,
                sample_batch_size=5,
                train_days=60,
                test_days=20
            )
            
            # 打印节点信息
            print_node_info(env_train)
            
            # 绘制节点地图
            plot_nodes_on_map(env_train, city=city, save_path=f"nodes_map_{city}_train.png")
            
            # 绘制订单热力图
            plot_heatmap_with_nodes(env_train, city=city, 
                                   save_path=f"nodes_heatmap_{city}_train.png", 
                                   is_train=True)
            
            # 分析训练集订单
            train_stats = analyze_orders(env_train, is_train=True)
            
            # 创建测试环境
            print("\n创建测试环境...")
            env_test = DroneTransferEnvLADE(
                env_args=env_args,
                algorithm="BR",
                city=city,
                is_train=False,
                device=device,
                sample_batch_size=5,
                train_days=60,
                test_days=20
            )
            
            # 分析测试集订单
            test_stats = analyze_orders(env_test, is_train=False)
            
            # 绘制测试集订单热力图
            plot_heatmap_with_nodes(env_test, city=city, 
                                   save_path=f"nodes_heatmap_{city}_test.png", 
                                   is_train=False)
            
            # 测试reset功能
            print("\n测试reset功能...")
            obs = env_train.reset()
            print(f"观察空间形状: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
            print(f"节点坐标形状: {env_train.nodes['coord'].shape}")
            print(f"订单from形状: {env_train.requests['from'].shape}")
            print(f"订单to形状: {env_train.requests['to'].shape}")
            print(f"可见订单数: {env_train.requests['visible'].sum().item()}")
            
            print(f"\n{city.upper()} 城市测试完成！")
            
        except Exception as e:
            print(f"测试 {city} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("所有测试完成！")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

