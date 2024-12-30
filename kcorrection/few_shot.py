import sys
import os
from models.models import few_shot, zero_shot
from models.plotting import plot_scatter
from models.train_utils import load_selected_columns_from_hdf5_chunks, calculate_few_shot
import numpy as np
import torch
from astropy.table import Table, join
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import pickle

TRAIN_DS_PATH = "data/kcorrs_train.hdf5"
TEST_DS_PATH =  "data/kcorrs_test.hdf5"

print("Loading data")

selected_columns = ["TARGETID", "Z", "KCORR01_SDSS_R", "KCORR01_SDSS_G", "KCORR01_SDSS_Z", "image_embeddings", "ABSMAG01_SDSS_R", "ABSMAG01_SDSS_G", "ABSMAG01_SDSS_Z"]

train_ds = load_selected_columns_from_hdf5_chunks(TRAIN_DS_PATH, selected_columns)
test_ds = load_selected_columns_from_hdf5_chunks(TEST_DS_PATH, selected_columns)
k_train, k_test = train_ds["KCORR01_SDSS_R"], test_ds["KCORR01_SDSS_R"]
z_train, z_test = train_ds["Z"], test_ds["Z"]

# Get data
data = {}
data["image"] = {}
X_train, X_test = (
    train_ds["image_embeddings"],
    test_ds["image_embeddings"],
)
embedding_scaler = StandardScaler().fit(X_train)
data["image"]["train"] = embedding_scaler.transform(X_train)
data["image"]["test"] = embedding_scaler.transform(X_test)

# Run predictions for multiple parameters and save results
results = {}

print("Calculating predictions")
for idx ,dims in enumerate([[32], [32,32], [32,32,32], [512], [512,512], [512,512,512], [512,256,126,32]]):
    for target_variable, bound in [
        ("KCORR01_SDSS_R", 0.37), 
        ("KCORR01_SDSS_G", None), 
        ("KCORR01_SDSS_Z", None), 
        ("ABSMAG01_SDSS_R", 0.37), 
        ("ABSMAG01_SDSS_G", None), 
        ("ABSMAG01_SDSS_Z", None)
    ]:
        preds_knn, k_test, stats = calculate_few_shot(
            train_ds, test_ds, z_train, z_test, data,
            column_name=target_variable, 
            mlp_hidden_dims = dims,
            redshift_upper_bound=bound
        )
        
        results[(idx, target_variable)] = {
            "predictions": preds_knn,
            "ground_truth": k_test,
            "stats": stats
        }

output_file = "results_few.pkl"
with open(output_file, "wb") as f:
    pickle.dump(results, f)

print(f"Results saved to {output_file}")