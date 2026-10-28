import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

# Read modeling dataset
df = pd.read_csv('ml-capstone26/data/modelling_table.csv')

# Group by hub_id and compute aggregate metrics
hub_profile = df.groupby('hub_id').agg(
    shipments=('is_delayed', 'count'),
    mean_delivery_hours=('delivery_hours', 'mean'),
    delay_rate=('is_delayed', 'mean'),
    mean_distance_km=('distance_km', 'mean'),
    mean_n_stops=('n_stops', 'mean')
).reset_index()

# Round the metrics for consistency
hub_profile['mean_delivery_hours'] = hub_profile['mean_delivery_hours'].round(3)
hub_profile['delay_rate'] = hub_profile['delay_rate'].round(3)
hub_profile['mean_distance_km'] = hub_profile['mean_distance_km'].round(3)
hub_profile['mean_n_stops'] = hub_profile['mean_n_stops'].round(3)

# Sort by ascending hub_id
hub_profile = hub_profile.sort_values(by='hub_id').reset_index(drop=True)

# Write to outputs/hub_profile.csv
os.makedirs('ml-capstone26/outputs', exist_ok=True)
hub_profile.to_csv('ml-capstone26/outputs/hub_profile.csv', index=False)

# Display the top rows of the generated hub profile
# display(hub_profile.head(20))

# Load & select cols and fit Scaler
# hub_profile = pd.read_csv('ml-capstone26/outputs/hub_profile.csv')
profile_cols = ['mean_delivery_hours', 'delay_rate', 'mean_distance_km', 'mean_n_stops']

scaler = StandardScaler()
X_profile_scaled = scaler.fit_transform(hub_profile[profile_cols])

# Fit KMeans with fixed parameters
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
hub_profile['cluster'] = kmeans.fit_predict(X_profile_scaled)
# display(hub_profile[['hub_id', 'cluster'] + profile_cols].head(10))

# Calculate pairwise Euclidean distances on the scaled matrix
n_hubs = len(hub_profile)
dist_matrix = squareform(pdist(X_profile_scaled, metric='euclidean'))

results = []

for K in [2, 3, 4, 5]:
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_profile_scaled)

    # Track sizes in descending order
    unique, counts = np.unique(labels, return_counts=True)
    sizes_desc = sorted(counts, reverse=True)
    cluster_sizes_str = "-".join(map(str, sizes_desc))

    same_cluster_dists = []
    diff_cluster_dists = []

    # Iterate through all unique unordered pairs
    for i in range(n_hubs):
        for j in range(i + 1, n_hubs):
            d = dist_matrix[i, j]
            if labels[i] == labels[j]:
                same_cluster_dists.append(d)
            else:
                diff_cluster_dists.append(d)

    mean_same = np.mean(same_cluster_dists) if same_cluster_dists else 0
    mean_diff = np.mean(diff_cluster_dists) if diff_cluster_dists else 0
    ratio = mean_diff / mean_same if mean_same > 0 else 0

    results.append({
        "K": K,
        "across_over_within": round(ratio, 3),
        "cluster_sizes": cluster_sizes_str
    })

# Build DataFrame and save
separation_df = pd.DataFrame(results)
separation_df.to_csv('ml-capstone26/outputs/part_d_separation.csv', index=False)
# display(separation_df)

# Calculate the mean delay_rate for each cluster to identify the sorting order
cluster_delay_means = hub_profile.groupby('cluster')['delay_rate'].mean().sort_values()

# Map sorted clusters: lowest -> Standard, middle -> Watch, highest -> Escalate
sorted_clusters = list(cluster_delay_means.index)
segment_mapping = {
    sorted_clusters[0]: 'Standard',
    sorted_clusters[1]: 'Watch',
    sorted_clusters[2]: 'Escalate'
}

hub_profile['segment'] = hub_profile['cluster'].map(segment_mapping)

# Merge with original hubs info to pull hub_city and hub_tier
hubs_info = pd.read_csv('ml-capstone26/data/hubs.csv')
hub_segments = hub_profile.merge(hubs_info[['hub_id', 'hub_city', 'hub_tier']], on='hub_id', how='left')

# Save outputs/hub_segments.csv
hub_segments_out = hub_segments[['hub_id', 'segment', 'hub_city', 'hub_tier']].sort_values(by='hub_id')
hub_segments_out.to_csv('ml-capstone26/outputs/hub_segments.csv', index=False)

# Create outputs/segment_profile.csv
segment_profile = hub_segments.groupby('segment').agg(
    hubs=('hub_id', 'count'),
    mean_delivery_hours=('mean_delivery_hours', 'mean'),
    delay_rate=('delay_rate', 'mean'),
    mean_distance_km=('mean_distance_km', 'mean'),
    mean_n_stops=('mean_n_stops', 'mean')
).reset_index()

# Maintain exact segment ordering: Standard, Watch, Escalate
segment_profile['segment'] = pd.Categorical(segment_profile['segment'], categories=['Standard', 'Watch', 'Escalate'], ordered=True)
segment_profile = segment_profile.sort_values('segment').reset_index(drop=True)

# Round columns for clean display
segment_profile['mean_delivery_hours'] = segment_profile['mean_delivery_hours'].round(3)
segment_profile['delay_rate'] = segment_profile['delay_rate'].round(3)
segment_profile['mean_distance_km'] = segment_profile['mean_distance_km'].round(3)
segment_profile['mean_n_stops'] = segment_profile['mean_n_stops'].round(3)

segment_profile.to_csv('ml-capstone26/outputs/segment_profile.csv', index=False)

# Create outputs/tier_vs_segment_mismatch.csv
tier_mapping = {
    1: 'Standard',
    2: 'Watch',
    3: 'Escalate'
}
hub_segments_out['expected_segment'] = hub_segments_out['hub_tier'].map(tier_mapping)
mismatch_df = hub_segments_out[hub_segments_out['segment'] != hub_segments_out['expected_segment']]
mismatch_df_out = mismatch_df[['hub_id', 'segment', 'hub_city', 'hub_tier']]

mismatch_df_out.to_csv('ml-capstone26/outputs/tier_vs_segment_mismatch.csv', index=False)

# Preview results
# print("Segment Profile:")
# display(segment_profile)
# print("\nMismatching Hubs:")
# display(mismatch_df_out)
