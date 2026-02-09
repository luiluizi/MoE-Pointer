#!/usr/bin/env python3
"""
generate_grids_and_heatmaps.py

读取 CSV（delivery 和 pickup），在给定经纬范围内按 500m x 500m 划分网格，
统计每格订单数，生成交互式 folium 地图（HTML）、静态热力 PNG、以及每格的统计 CSV。

输出文件（示例）：
 - delivery_grid_stats.csv
 - delivery_heatmap.html
 - delivery_heatmap.png
 - pickup_grid_stats.csv
 - pickup_heatmap.html
 - pickup_heatmap.png
"""

import os
import math
import numpy as np
import pandas as pd
import folium
from folium import Popup
from folium.plugins import HeatMap
import matplotlib.pyplot as plt

# ========== 用户可调整参数 ==========
# 文件路径
############################################## 杭州数据
# DELIVERY_CSV = "data/delivery/delivery_hz.csv"
# PICKUP_CSV = "data/pickup/pickup_hz.csv"
# LNG_MIN, LNG_MAX = 120.079320, 120.203163
# LAT_MIN, LAT_MAX = 30.243830, 30.350527


############################################## 上海数据
# DELIVERY_CSV = "data/delivery/delivery_sh.csv"
# PICKUP_CSV = "data/pickup/pickup_sh.csv"
# LNG_MIN, LNG_MAX = 121.424605, 121.550168
# LAT_MIN, LAT_MAX = 31.195720, 31.299977


############################################## 重庆数据
DELIVERY_CSV = "/mnt/jfs6/g-bairui/results/data/LADE/delivery/delivery_cq.csv"
PICKUP_CSV = "/mnt/jfs6/g-bairui/results/data/LADE/pickup/pickup_cq.csv"
LNG_MIN, LNG_MAX = 106.422522, 106.546171
LAT_MIN, LAT_MAX = 29.453564, 29.560914
# 网格尺寸（米）
GRID_M = 1000.0

# 输出目录
OUT_DIR = "/mnt/jfs6/g-bairui/results/data/output_heatmaps"
os.makedirs(OUT_DIR, exist_ok=True)
# ======================================

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
    if lng_edges[-1] < lng_max:
        lng_edges = np.append(lng_edges, lng_max)
    if lat_edges[-1] < lat_max:
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

def grid_stats_summary(counts):
    """
    统计：每格数量矩阵 -> 返回 DataFrame 格式（含 grid center 经纬、count）
    同时返回 overall_mean, overall_max
    """
    nrows, ncols = counts.shape
    counts_flat = counts.flatten()
    overall_mean = float(counts_flat.mean())
    overall_max = int(counts_flat.max())
    return overall_mean, overall_max

def counts_to_dataframe(counts, lng_edges, lat_edges):
    """
    把 counts 矩阵转换为 DataFrame，每行一个格子，包含:
      col_idx, row_idx, lng_min, lng_max, lat_min, lat_max, lng_center, lat_center, count
    row_idx 从 0（最南）到 nrows-1（最北）
    """
    nrows, ncols = counts.shape
    rows = []
    for r in range(nrows):
        for c in range(ncols):
            lat_min = lat_edges[r]
            lat_max = lat_edges[r+1]
            lng_min = lng_edges[c]
            lng_max = lng_edges[c+1]
            lat_c = (lat_min + lat_max) / 2.0
            lng_c = (lng_min + lng_max) / 2.0
            rows.append({
                "row_idx": r,
                "col_idx": c,
                "lng_min": lng_min,
                "lng_max": lng_max,
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lng_center": lng_c,
                "lat_center": lat_c,
                "count": int(counts[r, c])
            })
    df = pd.DataFrame(rows)
    return df

def make_folium_map(grid_df, counts, lng_edges, lat_edges, center_lat, center_lng, map_title, out_html):
    """
    使用 folium 生成交互式地图：
      - 每个网格用矩形着色（颜色按 count 映射）
      - 加入 HeatMap（点集为格中心 + 权重）
      - 保存为 out_html
    """
    # 创建基础地图
    m = folium.Map(location=[center_lat, center_lng], zoom_start=15, control_scale=True)
    # 颜色映射：简单线性（从白->红），使用 matplotlib 的 colormap
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    max_count = int(counts.max()) if counts.size > 0 else 0
    norm = colors.Normalize(vmin=0, vmax=max(1, max_count))
    cmap = cm.get_cmap('Reds')

    nrows, ncols = counts.shape

    # 添加矩形图层
    for idx, row in grid_df.iterrows():
        cnt = row['count']
        lat_min = row['lat_min']
        lat_max = row['lat_max']
        lng_min = row['lng_min']
        lng_max = row['lng_max']
        # colormap -> hex
        rgba = cmap(norm(cnt))
        hexcolor = colors.to_hex(rgba)
        # rectangle: bounds = [[south, west], [north, east]]
        bounds = [[lat_min, lng_min], [lat_max, lng_max]]
        popup = folium.Popup(f"count: {cnt}<br>cell: ({row['row_idx']},{row['col_idx']})", max_width=200)
        folium.Rectangle(
            bounds=bounds,
            color=None,
            fill=True,
            fill_color=hexcolor,
            fill_opacity=0.6 if cnt > 0 else 0.15,
            popup=popup
        ).add_to(m)

    # 添加 HeatMap（使用格子中心和权重）
    heat_data = [[r['lat_center'], r['lng_center'], r['count']] for _, r in grid_df.iterrows() if r['count'] > 0]
    if heat_data:
        HeatMap(heat_data, radius=25, max_zoom=16).add_to(m)

    # 添加边界框
    folium.Rectangle(bounds=[[lat_edges[0], lng_edges[0]], [lat_edges[-1], lng_edges[-1]]],
                     color='blue', fill=False).add_to(m)

    # 保存
    m.save(out_html)
    print(f"Saved folium map: {out_html}")

def make_static_heatmap_png(counts, out_png, title, lng_edges, lat_edges):
    """
    使用 matplotlib 绘制静态热力图（imshow），并保存为 PNG。
    我们让行从北到南在图像上是从上到下可视化，因此需要 flipud。
    """
    # 转换：counts 的行 0 是最南，imshow 想要上行是最大纬度 -> flipud
    arr = np.flipud(counts)  # now row 0 (top) is max latitude
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(arr, interpolation='nearest', aspect='equal')
    ax.set_title(title)
    ax.set_xlabel("网格列 (西->东)")
    ax.set_ylabel("网格行 (北->南)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('订单数')
    # 保存
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved static heatmap PNG: {out_png}")

def process_file(csv_path, lng_field_candidates, lat_field_candidates, prefix_name):
    """
    主处理函数：读取 csv，识别经纬字段（若字段名不同则从候选中找），
    生成 grid counts、保存 CSV、HTML、PNG，并返回总体统计。
    """
    if not os.path.exists(csv_path):
        print(f"Warning: file not found: {csv_path}. Skipping.")
        return None

    df = pd.read_csv(csv_path)
    # 查找经纬字段
    lng_col = None
    lat_col = None
    for c in lng_field_candidates:
        if c in df.columns:
            lng_col = c
            break
    for c in lat_field_candidates:
        if c in df.columns:
            lat_col = c
            break
    if lng_col is None or lat_col is None:
        print(f"Error: cannot find longitude/latitude columns in {csv_path}. Available columns: {df.columns.tolist()}")
        return None

    # 丢弃无效经纬
    df = df[[lng_col, lat_col]].copy()
    df = df.dropna(subset=[lng_col, lat_col])
    # 转为 numeric（防止字符串）
    df[lng_col] = pd.to_numeric(df[lng_col], errors='coerce')
    df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
    df = df.dropna(subset=[lng_col, lat_col])

    # 计算 delta 度数：用区域中点的纬度作为参考
    center_lat = (LAT_MIN + LAT_MAX) / 2.0
    delta_lng_deg, delta_lat_deg = meters_to_degrees(GRID_M, center_lat)

    # 构建网格边界
    lng_edges, lat_edges, ncols, nrows = build_grid(LNG_MIN, LNG_MAX, LAT_MIN, LAT_MAX, delta_lng_deg, delta_lat_deg)
    print(f"{prefix_name}: grid nrows={nrows}, ncols={ncols}, total_cells={nrows*ncols}")

    # 分配点到网格
    counts, row_idx, col_idx = assign_points_to_grid(df, lng_col, lat_col, lng_edges, lat_edges)

    # 统计
    overall_mean, overall_max = grid_stats_summary(counts)
    print(f"{prefix_name}: overall mean per cell = {overall_mean:.4f}, max per cell = {overall_max}")

    # 转 DataFrame 并保存
    grid_df = counts_to_dataframe(counts, lng_edges, lat_edges)
    out_csv = os.path.join(OUT_DIR, f"{prefix_name}_grid_stats.csv")
    grid_df.to_csv(out_csv, index=False)
    print(f"Saved grid stats CSV: {out_csv}")

    # folium map
    out_html = os.path.join(OUT_DIR, f"{prefix_name}_heatmap.html")
    make_folium_map(grid_df, counts, lng_edges, lat_edges, center_lat=center_lat, center_lng=(LNG_MIN+LNG_MAX)/2.0,
                    map_title=f"{prefix_name} heatmap", out_html=out_html)

    # static PNG heatmap
    out_png = os.path.join(OUT_DIR, f"{prefix_name}_heatmap.png")
    make_static_heatmap_png(counts, out_png, title=f"{prefix_name} Grid Heatmap", lng_edges=lng_edges, lat_edges=lat_edges)

    return {
        "prefix": prefix_name,
        "nrows": nrows,
        "ncols": ncols,
        "total_cells": nrows * ncols,
        "mean_per_cell": overall_mean,
        "max_per_cell": overall_max,
        "grid_csv": out_csv,
        "map_html": out_html,
        "map_png": out_png
    }

def main():
    # 对 delivery 和 pickup 分别处理，字段名候选以防 CSV 字段命名不同
    delivery_result = process_file(
        DELIVERY_CSV,
        lng_field_candidates=["delivery_gps_lng", "lng", "longitude", "lon"],
        lat_field_candidates=["delivery_gps_lat", "lat", "latitude"]
        ,
        prefix_name="delivery"
    )
    pickup_result = process_file(
        PICKUP_CSV,
        lng_field_candidates=["delivery_gps_lng", "pickup_gps_lng", "lng", "longitude", "lon"],
        lat_field_candidates=["delivery_gps_lat", "pickup_gps_lat", "lat", "latitude"],
        prefix_name="pickup"
    )

    # 汇总打印
    for res in (delivery_result, pickup_result):
        if res is None:
            continue
        print("======")
        print(f"{res['prefix']} summary:")
        print(f" grid: {res['nrows']} rows x {res['ncols']} cols = {res['total_cells']} cells")
        print(f" mean per cell: {res['mean_per_cell']:.4f}")
        print(f" max in one cell: {res['max_per_cell']}")
        print(f" grid csv: {res['grid_csv']}")
        print(f" interactive map html: {res['map_html']}")
        print(f" static heatmap png: {res['map_png']}")
    print("Done.")

if __name__ == "__main__":
    main()