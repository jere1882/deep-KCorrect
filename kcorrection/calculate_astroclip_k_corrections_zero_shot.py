import pandas as pd
import numpy as np
import pickle
from utils.train_utils import calculate_zero_shot
from utils.data_utils import prepare_dataset

TRAIN_DS_PATH = "data/train_df_kcorr_panda.h5"
TEST_DS_PATH =  "data/test_df_kcorr_panda.h5"

train_df = pd.read_hdf(TRAIN_DS_PATH, key='data')
test_df = pd.read_hdf(TEST_DS_PATH, key='data')

TARGET_VARIABLES = [
    "KCORR01_SDSS_R", 
    "KCORR01_SDSS_G", 
    "KCORR01_SDSS_Z", 
    "ABSMAG01_SDSS_R", 
    "ABSMAG01_SDSS_G", 
    "ABSMAG01_SDSS_Z"
]

NEAREST_NEIGHBORS = [1, 3, 10, 30, 64, 100]
MIN_REDSHIFT = 0
MAX_REDSHIFT = 2
# Block 1: Calculate predictions across all redshifts for each target band
results_across_all_redshifts = {}

print("Block 1: Calculating predictions across all redshifts")
for n_neighbors in NEAREST_NEIGHBORS:
    for target_variable in TARGET_VARIABLES:

        print(f"Processing {target_variable} with neighbors {n_neighbors}")

        X_train, y_train, X_test, y_test = (
            prepare_dataset(train_df, test_df, MIN_REDSHIFT, MAX_REDSHIFT, target_variable)
        )

        results_across_all_redshifts[(n_neighbors, target_variable)] = (
            calculate_zero_shot(X_train, y_train, X_test, y_test, n_neighbors = n_neighbors)
        )



print("Block 2: Calculating predictions for each pair of bands")
results_pairwise = {}

threshold_r = 0.3885
threshold_g1 = 0.3031
threshold_g2 = 0.814

VALID_PAIRS = [
    ("K_G_G", "KCORR01_SDSS_G", MIN_REDSHIFT, threshold_g1),
    ("K_G_R", "KCORR01_SDSS_G", threshold_g1, threshold_g2),
    ("K_G_Z", "KCORR01_SDSS_G", threshold_g2, MAX_REDSHIFT),
    ("K_R_R", "KCORR01_SDSS_R", MIN_REDSHIFT, threshold_r),
    ("K_R_G", "KCORR01_SDSS_R", threshold_r, MAX_REDSHIFT),
    ("K_Z_Z", "KCORR01_SDSS_Z", MIN_REDSHIFT, MAX_REDSHIFT),
    ("absmag_G_G", "ABSMAG01_SDSS_G", MIN_REDSHIFT, threshold_g1),
    ("absmag_G_R", "ABSMAG01_SDSS_G", threshold_g1, threshold_g2),
    ("absmag_G_Z", "ABSMAG01_SDSS_G", threshold_g2, MAX_REDSHIFT),
    ("absmag_R_R", "ABSMAG01_SDSS_R", MIN_REDSHIFT, threshold_r),
    ("absmag_R_G", "ABSMAG01_SDSS_R", threshold_r, MAX_REDSHIFT),
    ("absmag_Z_Z", "ABSMAG01_SDSS_Z", MIN_REDSHIFT, MAX_REDSHIFT),
]

for n_neighbors in [1, 3, 10, 30, 64, 100]:
    for pair in VALID_PAIRS:
        print(f"Processing {pair[0]} with neighbors {n_neighbors}")

        X_train, y_train, X_test, y_test = (
            prepare_dataset(train_df, test_df, pair[2], pair[3], pair[1])
        )

        results_pairwise[(n_neighbors, pair[0])] = (
            calculate_zero_shot(X_train, y_train, X_test, y_test, n_neighbors = n_neighbors)
        )


print("Done making predictions")

output_file = "data/astroclip_predictions_zero_shot.pkl"
results = [results_across_all_redshifts, results_pairwise]
with open(output_file, "wb") as f:
    pickle.dump(results, f)

print(f"Results saved to {output_file}")