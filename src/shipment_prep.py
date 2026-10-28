import numpy as np
import pandas as pd

# CONSTANTS
PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]

def clean_shipments(raw, medians=None):
  clean_df = raw.drop_duplicates().copy()

  # clean dataset
  clean_df["service_type"] = clean_df["service_type"].str.strip().str.title()
  clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")
  clean_df["declared_value_inr"] = clean_df["declared_value"].str.replace(r'[^\d.]', '', regex=True).astype(float)
  clean_df["dispatch_time"] = pd.to_datetime(clean_df["dispatch_time"])
  clean_df.loc[clean_df["parcel_weight_kg"] > 40, "parcel_weight_kg"] = np.nan

  # compute medians
  if medians is None:
    medians = {
        "distance_km": clean_df["distance_km"].median(),
        "parcel_weight_kg": clean_df["parcel_weight_kg"].median(),
        "declared_value_inr": clean_df["declared_value_inr"].median()
    }

  # fill values in dataset
  clean_df["distance_km"] = clean_df["distance_km"].fillna(medians["distance_km"])
  clean_df["parcel_weight_kg"] = clean_df["parcel_weight_kg"].fillna(medians["parcel_weight_kg"])
  clean_df["declared_value_inr"] = clean_df["declared_value_inr"].fillna(medians["declared_value_inr"])

  return clean_df, medians

  def build_features(clean, hubs):
    hubs_df = hubs.copy()
    clean_df = clean.copy()

  merged_df = clean_df.merge(hubs_df, on="hub_id", how="left")

  # Compute new columns on the merged shipment dataset
  merged_df["promised_hours"] = merged_df["service_type"].map(PROMISED_HOURS).astype(int)
  merged_df["is_cod"] = (merged_df["payment_mode"] == "COD").astype(int)
  merged_df["is_weekend"] = (merged_df["dispatch_time"].dt.dayofweek >= 5).astype(int)
  merged_df["dispatch_hour"] = merged_df["dispatch_time"].dt.hour.astype(int)
  merged_df["log_declared_value"] = np.log(merged_df["declared_value_inr"])

  return merged_df
