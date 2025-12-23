"""
train_clustering_pipeline.py

Unsupervised clustering on REAL Bengaluru housing data
augmented with infrastructure features.

Clusters are INTERPRETED as Affordable / Moderate / Luxury
based primarily on price, refined by location & infrastructure.
"""

# ---------------------------------------------------
# IMPORTS
# ---------------------------------------------------
import os
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------
# MAIN TRAINING FUNCTION
# ---------------------------------------------------
def train_clustering_model(
    input_csv: str,
    output_dir: str,
    n_clusters: int = 3,
    random_state: int = 42
):

    print("=" * 60)
    print("STARTING REAL ESTATE CLUSTERING PIPELINE")
    print("=" * 60)

    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)

    # ---------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------
    df = pd.read_csv(input_csv)
    print(f"Loaded dataset with {len(df)} rows")

    # ---------------------------------------------------
    # REQUIRED COLUMNS CHECK
    # ---------------------------------------------------
    required_cols = [
        "price_inr",
        "airport_distance_km",
        "railway_station_distance_km",
        "nearest_hospital_distance_km",
        "hospitals_count",
        "schools_count",
        "location"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ---------------------------------------------------
    # HANDLE LOCATION (ENCODING)
    # ---------------------------------------------------
    le = LabelEncoder()
    df["location_encoded"] = le.fit_transform(df["location"].astype(str))

    # ---------------------------------------------------
    # SELECT FEATURES FOR CLUSTERING
    # ---------------------------------------------------
    feature_cols = [
        "price_inr",
        "airport_distance_km",
        "railway_station_distance_km",
        "nearest_hospital_distance_km",
        "hospitals_count",
        "schools_count",
        "location_encoded"
    ]

    X = df[feature_cols].copy()

    # ---------------------------------------------------
    # LOG TRANSFORM PRICE (CRITICAL)
    # ---------------------------------------------------
    X["price_inr"] = np.log1p(X["price_inr"])

    # ---------------------------------------------------
    # HANDLE MISSING VALUES
    # ---------------------------------------------------
    X = X.fillna(X.median())

    # ---------------------------------------------------
    # SCALE FEATURES
    # ---------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------
    # KMEANS CLUSTERING
    # ---------------------------------------------------
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=25,
        max_iter=500
    )

    clusters = kmeans.fit_predict(X_scaled)
    df["cluster"] = clusters

    # ---------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------
    db_index = davies_bouldin_score(X_scaled, clusters)
    print(f"Davies–Bouldin Index: {db_index:.4f}")

    with open(os.path.join(output_dir, "outputs", "model_evaluation.txt"), "w") as f:
        f.write(f"Davies–Bouldin Index: {db_index:.4f}\n")

    # ---------------------------------------------------
    # CLUSTER INTERPRETATION (PRICE-BASED)
    # ---------------------------------------------------
    price_median = (
        df.groupby("cluster")["price_inr"]
        .median()
        .sort_values()
    )

    labels = ["Affordable", "Moderate", "Luxury"]
    cluster_label_map = {
        cluster_id: labels[i]
        for i, cluster_id in enumerate(price_median.index)
    }

    df["cluster_label"] = df["cluster"].map(cluster_label_map)

    print("Cluster mapping:")
    for k, v in cluster_label_map.items():
        print(f"Cluster {k} → {v}")

    # ---------------------------------------------------
    # SAVE MODEL
    # ---------------------------------------------------
    model_bundle = {
        "kmeans": kmeans,
        "scaler": scaler,
        "features": feature_cols,
        "cluster_label_map": cluster_label_map,
        "location_encoder": le
    }

    joblib.dump(
        model_bundle,
        os.path.join(output_dir, "models", "kmeans_model.joblib")
    )

    # ---------------------------------------------------
    # PCA VISUALIZATION
    # ---------------------------------------------------
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    for cid in np.unique(clusters):
        mask = clusters == cid
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=cluster_label_map[cid],
            alpha=0.6,
            s=15
        )

    plt.title("PCA Projection of Housing Clusters")
    plt.legend()
    plt.grid(True)

    pca_path = os.path.join(output_dir, "outputs", "pca_clusters.png")
    plt.savefig(pca_path, bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------
    # SAVE CLUSTERED DATA
    # ---------------------------------------------------
    out_csv = os.path.join(output_dir, "outputs", "clustered_bengaluru.csv")
    df.to_csv(out_csv, index=False)

    print("Pipeline completed successfully.")
    print("=" * 60)


# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/bengaluru_housing_augmented_full.csv"
    )
    parser.add_argument(
        "--out_dir",
        default="."
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=3
    )

    args = parser.parse_args()

    train_clustering_model(
        input_csv=args.input,
        output_dir=args.out_dir,
        n_clusters=args.clusters
    )










# """
# train_clustering_pipeline.py

# This script trains an UNSUPERVISED KMeans clustering model
# on the Bengaluru housing dataset generated by generate_data.py.

# ACADEMIC INTENT:
# ----------------
# This file is intentionally written in a detailed and verbose manner
# to make it suitable for a 3rd-year B.Tech / B.E. / BCA / BSc submission.

# WHAT THIS SCRIPT DOES:
# ----------------------
# 1. Loads the synthetic housing dataset
# 2. Selects numeric features (no income, no affordability)
# 3. Handles missing values using MEDIAN IMPUTATION
# 4. Applies log transformation to skewed features
# 5. Scales features using StandardScaler
# 6. Uses KMeans clustering (k = 3)
# 7. Evaluates clustering using Davies–Bouldin Index
# 8. Visualizes clusters using PCA
# 9. Assigns human-readable cluster labels
# 10. Saves trained model and outputs for Streamlit app

# IMPORTANT:
# ----------
# - This is a FULLY UNSUPERVISED ML pipeline
# - No labels are used during training
# - Cluster meaning is INTERPRETED AFTER training
# """
# # -------------------------------------------------------------------
# # FIX PYTHON PATH FOR PROJECT STRUCTURE
# # -------------------------------------------------------------------
# import sys
# import os

# sys.path.append(
#     os.path.abspath(
#         os.path.join(os.path.dirname(__file__), "..")
#     )
# )

# # -------------------------------------------------------------------
# # IMPORT REQUIRED LIBRARIES
# # -------------------------------------------------------------------

# import argparse
# import joblib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.metrics import davies_bouldin_score

# # Import preprocessing configuration
# from utils.preprocess import NUM_COLS, preprocess_features

# # -------------------------------------------------------------------
# # MAIN TRAINING FUNCTION
# # -------------------------------------------------------------------
# def train_clustering_model(
#     input_csv: str,
#     output_dir: str,
#     n_clusters: int = 3,
#     random_state: int = 42
# ):
#     """
#     Trains KMeans clustering model on housing dataset.

#     Parameters:
#     -----------
#     input_csv : str
#         Path to housing dataset CSV
#     output_dir : str
#         Directory to save model and outputs
#     n_clusters : int
#         Number of clusters (default = 3)
#     random_state : int
#         Random seed for reproducibility
#     """

#     print("=" * 60)
#     print("STARTING HOUSING CLUSTERING PIPELINE")
#     print("=" * 60)

#     # ---------------------------------------------------------------
#     # CREATE REQUIRED DIRECTORIES
#     # ---------------------------------------------------------------
#     models_dir = os.path.join(output_dir, "models")
#     outputs_dir = os.path.join(output_dir, "outputs")

#     os.makedirs(models_dir, exist_ok=True)
#     os.makedirs(outputs_dir, exist_ok=True)

#     # ---------------------------------------------------------------
#     # LOAD DATASET
#     # ---------------------------------------------------------------
#     print("\n[1] Loading dataset...")
#     df = pd.read_csv(input_csv)

#     print(f"Dataset loaded successfully with {len(df)} rows")
#     print("Columns available:")
#     print(list(df.columns))

#     # ---------------------------------------------------------------
#     # VALIDATE REQUIRED NUMERIC COLUMNS
#     # ---------------------------------------------------------------
#     print("\n[2] Validating required numeric columns...")
#     missing_cols = [col for col in NUM_COLS if col not in df.columns]

#     if missing_cols:
#         raise ValueError(
#             f"Missing required columns in dataset: {missing_cols}"
#         )

#     print("All required numeric columns are present.")

#     # ---------------------------------------------------------------
#     # PREPROCESS DATA
#     # ---------------------------------------------------------------
#     print("\n[3] Preprocessing features...")
#     X_processed = preprocess_features(df)

#     print("Preprocessing completed.")
#     print("Shape after preprocessing:", X_processed.shape)

#     # ---------------------------------------------------------------
#     # FEATURE SCALING
#     # ---------------------------------------------------------------
#     print("\n[4] Scaling features using StandardScaler...")
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_processed)

#     print("Feature scaling completed.")

#     # ---------------------------------------------------------------
#     # TRAIN KMEANS MODEL
#     # ---------------------------------------------------------------
#     print("\n[5] Training KMeans clustering model...")
#     kmeans = KMeans(
#         n_clusters=n_clusters,
#         random_state=random_state,
#         n_init=20,
#         max_iter=500
#     )

#     cluster_labels = kmeans.fit_predict(X_scaled)
#     df["cluster"] = cluster_labels

#     print(f"KMeans training completed with {n_clusters} clusters.")

#     # ---------------------------------------------------------------
#     # EVALUATE CLUSTER QUALITY
#     # ---------------------------------------------------------------
#     print("\n[6] Evaluating clustering using Davies–Bouldin Index...")
#     db_index = davies_bouldin_score(X_scaled, cluster_labels)

#     print(f"Davies–Bouldin Index: {db_index:.4f}")
#     with open("outputs/model_evaluation.txt", "w") as f:
#         f.write("Model Evaluation Metrics\n")
#         f.write("========================\n")
#         f.write(f"Davies–Bouldin Index: {db_index:.4f}\n")
#         f.write("Lower value indicates better clustering.\n")

#     print("(Lower value indicates better clustering)")

#     # ---------------------------------------------------------------
#     # INTERPRET CLUSTERS USING MEDIAN PRICE
#     # ---------------------------------------------------------------
#     print("\n[7] Interpreting clusters (post-training)...")

#     cluster_price_median = (
#         df.groupby("cluster")["median_price"]
#         .median()
#         .sort_values()
#     )

#     cluster_label_map = {}
#     readable_labels = ["Affordable", "Moderate", "Luxury"]

#     for idx, cluster_id in enumerate(cluster_price_median.index):
#         cluster_label_map[int(cluster_id)] = readable_labels[idx]

#     print("Cluster interpretation:")
#     for k, v in cluster_label_map.items():
#         print(f"  Cluster {k} → {v}")

#     # ---------------------------------------------------------------
#     # SAVE TRAINED MODEL BUNDLE
#     # ---------------------------------------------------------------
#     print("\n[8] Saving trained model...")

#     model_bundle = {
#         "kmeans": kmeans,
#         "scaler": scaler,
#         "cols": NUM_COLS,
#         "cluster_label_map": cluster_label_map
#     }

#     model_path = os.path.join(models_dir, "kmeans_model.joblib")
#     joblib.dump(model_bundle, model_path)

#     print(f"Model saved to: {model_path}")

#     # ---------------------------------------------------------------
#     # PCA VISUALIZATION (2D)
#     # ---------------------------------------------------------------
#     print("\n[9] Creating PCA visualization...")

#     pca = PCA(n_components=2, random_state=random_state)
#     X_pca = pca.fit_transform(X_scaled)

#     plt.figure(figsize=(8, 6))
#     for cluster_id in np.unique(cluster_labels):
#         mask = cluster_labels == cluster_id
#         plt.scatter(
#             X_pca[mask, 0],
#             X_pca[mask, 1],
#             label=f"Cluster {cluster_id}",
#             alpha=0.6,
#             s=20
#         )

#     plt.title("PCA Projection of Housing Clusters")
#     plt.xlabel("Principal Component 1")
#     plt.ylabel("Principal Component 2")
#     plt.legend()
#     plt.grid(True)

#     pca_path = os.path.join(outputs_dir, "pca_clusters.png")
#     plt.savefig(pca_path, bbox_inches="tight")
#     plt.close()

#     print(f"PCA plot saved to: {pca_path}")

#     # ---------------------------------------------------------------
#     # SAVE CLUSTERED DATASET
#     # ---------------------------------------------------------------
#     print("\n[10] Saving clustered dataset...")

#     clustered_csv_path = os.path.join(
#         outputs_dir, "clustered_bengaluru.csv"
#     )
#     df.to_csv(clustered_csv_path, index=False)

#     print(f"Clustered dataset saved to: {clustered_csv_path}")

#     # ---------------------------------------------------------------
#     # SAVE CLUSTER SUMMARY
#     # ---------------------------------------------------------------
#     print("\n[11] Generating cluster summary...")

#     cluster_summary = df.groupby("cluster").agg({
#         "median_price": "median",
#         "median_rent": "median",
#         "population_density": "median",
#         "schools_count": "median",
#         "hospitals_count": "median",
#         "area_num": "count"
#     }).rename(columns={"area_num": "total_areas"}).reset_index()

#     summary_path = os.path.join(
#         outputs_dir, "cluster_summary.csv"
#     )
#     cluster_summary.to_csv(summary_path, index=False)

#     print(f"Cluster summary saved to: {summary_path}")

#     print("\n" + "=" * 60)
#     print("CLUSTERING PIPELINE COMPLETED SUCCESSFULLY")
#     print("=" * 60)


# # -------------------------------------------------------------------
# # SCRIPT ENTRY POINT
# # -------------------------------------------------------------------
# if __name__ == "__main__":

#     parser = argparse.ArgumentParser(
#         description="Train unsupervised housing clustering model"
#     )

#     parser.add_argument(
#         "--input",
#         type=str,
#         default="data/housing_bengaluru_realistic.csv",
#         help="Path to input housing CSV"
#     )

#     parser.add_argument(
#         "--out_dir",
#         type=str,
#         default=".",
#         help="Base output directory"
#     )

#     parser.add_argument(
#         "--clusters",
#         type=int,
#         default=3,
#         help="Number of clusters (default = 3)"
#     )

#     args = parser.parse_args()

#     train_clustering_model(
#         input_csv=args.input,
#         output_dir=args.out_dir,
#         n_clusters=args.clusters
#     )
