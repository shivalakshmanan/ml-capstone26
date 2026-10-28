import pandas as pd
import numpy as np

# Read input data
df = pd.read_csv("ml-capstone26/data/modelling_table.csv")

bins = [0, 200, 400, 600, 2000]
labels = ["0-200", "201-400", "401-600", "601+"]
df["distance_band"] = pd.cut(df["distance_km"], bins=bins, labels=labels, right=True)

# Define dimensions to analyze
dimensions = ["service_type", "hub_tier", "distance_band"]
all_groups = []

for dim in dimensions:
    # compute aggregates
    grouped = df.groupby(dim, observed=False).agg(
        shipments=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
        delay_rate=("is_delayed", "mean"),
        mean_hours=("delivery_hours", "mean")
    ).reset_index()

    # Standardize columns
    grouped.rename(columns={dim: "group"}, inplace=True)
    grouped.insert(0, "dimension", dim)
    all_groups.append(grouped)

# format coulmns
delay_by_group = pd.concat(all_groups, ignore_index=True)
delay_by_group["group"] = delay_by_group["group"].astype(str)
delay_by_group["shipments"] = delay_by_group["shipments"].astype(int)
delay_by_group["delayed"] = delay_by_group["delayed"].astype(int)
delay_by_group["delay_rate"] = delay_by_group["delay_rate"].round(4)
delay_by_group["mean_hours"] = delay_by_group["mean_hours"].round(4)


delay_by_group.to_csv("ml-capstone26/outputs/part_a_delay_by_group.csv", index=False)
# display(delay_by_group)
