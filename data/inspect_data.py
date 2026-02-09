import geohash
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--place", type=str, default="tw")
parser.add_argument("--heatmap", action="store_true", default=False)
args = parser.parse_args()

place = args.place
delta = 0.02197265625

config = {
    "tw": {
        "n_node": 36,
        "except": [],
    },
    "se": {
        "n_node": 36,
        "except": ['u7xqm']
    },
    "sg": {
        "n_node": 36,
        "except": []
    }
}[place]

orders = pd.read_csv(f"data/data_{place}/orders_{place}_train.txt", index_col=0)
vendors = pd.read_csv(f"data/data_{place}/vendors_{place}.txt", index_col=0)

n_node = config["n_node"]
except_node = config["except"]

vendors_geohash = vendors.geohash.value_counts().index
nodes = orders.geohash.value_counts().drop(except_node).drop(vendors_geohash)[:n_node - len(vendors_geohash)].index.tolist() + vendors_geohash.tolist()
assert n_node == len(nodes)

nodes_coords = np.array([geohash.decode(code) for code in nodes])
for code in nodes:
    _, _, dlat, dlon = geohash.decode(code, True)
    assert dlat == delta and dlon == delta


dist = (np.abs(nodes_coords[:, None] - nodes_coords[None]).sum(-1) / delta + 0.5).astype(int)
assert (dist % 2 == 0).all()
dist = dist // 2

orders = orders.join(vendors[["vendor_id", "geohash"]].set_index("vendor_id"), on="vendor_id", how="left", rsuffix="_v")
assert vendors.geohash.isin(nodes).all()

orders = orders.drop(orders.index[~orders.geohash.isin(nodes)])
orders = orders.drop(orders.index[orders.geohash==orders.geohash_v])
orders = orders.reset_index()


orders["order_time"] = pd.to_datetime(orders["order_time"], format="%H:%M:%S")
orders["from"] = orders["geohash_v"].map({code: idx for idx, code in enumerate(nodes)})
orders["to"] = orders["geohash"].map({code: idx for idx, code in enumerate(nodes)})

global_od = orders.pivot_table(index="from", columns="to", aggfunc="size", fill_value=0)
global_od = global_od.reindex(index=range(n_node), columns=range(n_node), fill_value=0)
global_od = global_od.to_numpy()

if args.heatmap:
    start_count = orders["from"].value_counts()
    end_count = orders["to"].value_counts()
    heatmap_df = pd.concat([end_count, start_count], axis=1, keys=["start_count", "end_count"]).fillna(0)
    heatmap_df[["lat", "lon"]] = nodes_coords[heatmap_df.index]
    heatmap_df["geohash"] = [nodes[i] for i in heatmap_df.index]
    import folium
    from folium.plugins import HeatMap
    def add_geohash_polygon(map_obj, geo, color='blue', fill_color='blue', fill_opacity=0.0):
        bounds = geohash.bbox(geo)
        south, north = bounds['s'], bounds['n']
        west, east = bounds['w'], bounds['e']
        
        polygon_points = [
            [south, west],
            [south, east],
            [north, east],
            [north, west]
        ]
        
        folium.Polygon(locations=polygon_points, color=color, weight=0.1, fill_color=fill_color, fill_opacity=fill_opacity).add_to(map_obj)
        
    def create_heatmap(df: pd.DataFrame):
        df["ratio_start"] = df["start_count"] / df["start_count"].max()
        df["ratio_end"] = df["end_count"] / df["end_count"].max()
        df["rank_start"] = df["start_count"].rank() / len(df)
        df["rank_end"] = df["end_count"].rank() / len(df)
        m = folium.Map(location=[df['lat'].mean(), df['lon'].mean()], zoom_start=12)
        heat_data_start = df[['lat', 'lon', 'rank_start']].values.tolist()
        heat_data_end = df[['lat', 'lon', 'rank_end']].values.tolist()
        gradiant = {
            "0.2": "blue",
            "0.4": "cyan",
            "0.6": "lime",
            "0.8": "yellow",
            "1": "red",
        }
        gradiant = {str(key): value for key, value in gradiant.items()}
        # gradiant = None
        print(df[['lat', 'lon', 'start_count', "ratio_start", 'end_count', "ratio_end"]])
        HeatMap(heat_data_start, radius=70, blur=40, gradient=gradiant, min_opacity=0.3).add_to(folium.FeatureGroup(name='Start Points').add_to(m))
        HeatMap(heat_data_end, radius=70, blur=40, gradient=gradiant, min_opacity=0.3).add_to(folium.FeatureGroup(name='End Points').add_to(m))
        folium.LayerControl().add_to(m)

        for _, row in df.iterrows():
            add_geohash_polygon(m, row["geohash"])
        return m
    print(heatmap_df)
    m = create_heatmap(heatmap_df)
    m.save(f"heatmap_{place}.html")

hour_od = {}
for hour, hour_df in orders.groupby(orders["order_time"].dt.hour):
    od = hour_df.pivot_table(index="from", columns="to", aggfunc="size", fill_value=0)
    od = od.reindex(index=range(n_node), columns=range(n_node), fill_value=0)
    od = od.to_numpy()
    hour_od[hour] = od

train_orders = orders

orders = pd.read_csv(f"data/data_{place}/orders_{place}_test.txt", index_col=0)
orders = orders.join(vendors[["vendor_id", "geohash"]].set_index("vendor_id"), on="vendor_id", how="left", rsuffix="_v")
assert vendors.geohash.isin(nodes).all()

orders = orders.drop(orders.index[~orders.geohash.isin(nodes)])
orders = orders.drop(orders.index[orders.geohash==orders.geohash_v])
orders = orders.reset_index()


orders["order_time"] = pd.to_datetime(orders["order_time"], format="%H:%M:%S")
orders["from"] = orders["geohash_v"].map({code: idx for idx, code in enumerate(nodes)})
orders["to"] = orders["geohash"].map({code: idx for idx, code in enumerate(nodes)})


for order_day, day_df in orders.groupby("order_day"):
    for hour, hour_df in day_df.groupby(day_df["order_time"].dt.hour):
        od = hour_df.pivot_table(index="from", columns="to", aggfunc="size", fill_value=0)
        od = od.reindex(index=range(n_node), columns=range(n_node), fill_value=0)
        od = od.to_numpy()
        train_od = hour_od[hour]
        print(order_day, hour, len(hour_df), np.sum(od * train_od) / np.linalg.norm(od) / np.linalg.norm(train_od))

# for paper usage:
orders = train_orders
ODs = []
for order_day, day_df in orders.groupby("order_day"):
    od = day_df.pivot_table(index="from", columns="to", aggfunc="size", fill_value=0)
    od = od.reindex(index=range(n_node), columns=range(n_node), fill_value=0)
    od = od.to_numpy()
    ODs.append(od)
OD = np.stack(ODs, axis=0).mean(axis=0)
similarities = []
for order_day, day_df in orders.groupby("order_day"):
    od = day_df.pivot_table(index="from", columns="to", aggfunc="size", fill_value=0)
    od = od.reindex(index=range(n_node), columns=range(n_node), fill_value=0)
    od = od.to_numpy()
    sim = np.sum(od * OD) / np.linalg.norm(od) / np.linalg.norm(OD)
    print(order_day, len(day_df), sim)
    similarities.append(sim)
print("mean similarities:", np.mean(similarities))