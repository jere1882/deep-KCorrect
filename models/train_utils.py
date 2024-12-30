import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from models.models import few_shot, zero_shot, zero_shot_with_uncertainty
from models.plotting import plot_scatter

def calculate_zero_shot(
    train_ds, test_ds, z_train, z_test, data, column_name="KCORR01_SDSS_R", n_neighbors = 64, redshift_upper_bound = None, plot=False):
    # Get retrieve_k_corrections
    k_train = train_ds[column_name]
    k_test = test_ds[column_name]
    z_train = train_ds["Z"]
    z_test = test_ds["Z"]

    train_data = data["image"]["train"]
    test_data = data["image"]["test"]

    if redshift_upper_bound != None:
        cond_tr = z_train <= redshift_upper_bound
        k_train = k_train[cond_tr]
        train_data = train_data[cond_tr]

        cond_test = z_test <= redshift_upper_bound
        k_test = k_test[cond_test]
        test_data = test_data[cond_test]
        z_test = z_test[cond_test]
        
    # Scale properties
    scaler = {"mean": k_train.mean(), "std": k_train.std()}
    k_train = (k_train - scaler["mean"]) / scaler["std"]
    
    preds_knn = {}
    raw_preds_knn = zero_shot(train_data, k_train, test_data, n_neighbors)
    preds_knn["astroclip_image"] = raw_preds_knn * scaler["std"] + scaler["mean"]
    preds_knn["AstroCLIP"] = preds_knn.pop("astroclip_image")

    if plot:
        plot_scatter(preds_knn, k_test)

    mae_knn = np.mean(np.abs(preds_knn['AstroCLIP'] - k_test))
    r2_knn = r2_score(k_test, preds_knn['AstroCLIP'])

    stats = {
        "mae_knn" : mae_knn,
        "r2_knn" : r2_knn,
        "z" : z_test
    }

    return preds_knn, k_test, stats


def load_selected_columns_from_hdf5_chunks(hdf5_path, selected_columns):
    """Load specific columns from a structured dataset in an HDF5 file with chunks."""
    column_data = {col: [] for col in selected_columns}  # Initialize storage for all columns
    
    with h5py.File(hdf5_path, 'r') as f:
        if 'chunks' not in f:
            raise KeyError("The file does not contain a 'chunks' group.")
        
        chunks_group = f['chunks']  # Access the chunks group
        
        for chunk_name in chunks_group:
            chunk_data = chunks_group[chunk_name]
            
            # Check if the dataset in the chunk is structured
            if not isinstance(chunk_data, h5py.Dataset):
                print(f"Skipping {chunk_name}: not a dataset.")
                continue
            
            # Extract selected columns from this chunk
            for column in selected_columns:
                if column in chunk_data.dtype.names:  # Verify column exists in this chunk
                    column_data[column].extend(chunk_data[column][:])  # Append data
                else:
                    print(f"Warning: Column '{column}' not found in chunk {chunk_name}.")
    
        # Convert lists to arrays
        column_data = {col: np.array(data) for col, data in column_data.items()}
        return column_data

def calculate_few_shot(
    train_ds, test_ds, z_train, z_test, data, column_name="KCORR01_SDSS_R", mlp_hidden_dims = [32], redshift_upper_bound = None, plot=False):
    # Get retrieve_k_corrections
    k_train = train_ds[column_name]
    k_test = test_ds[column_name]
    z_train = train_ds["Z"]
    z_test = test_ds["Z"]

    train_data = data["image"]["train"]
    test_data = data["image"]["test"]

    if redshift_upper_bound != None:
        cond_tr = z_train <= redshift_upper_bound
        k_train = k_train[cond_tr]
        train_data = train_data[cond_tr]

        cond_test = z_test <= redshift_upper_bound
        k_test = k_test[cond_test]
        test_data = test_data[cond_test]
        
    # Scale properties
    scaler = {"mean": k_train.mean(), "std": k_train.std()}
    k_train = (k_train - scaler["mean"]) / scaler["std"]
    
    # Perfrom knn and mlp
    preds_knn, preds_mlp = {}, {}
    raw_preds_mlp = few_shot(train_data, k_train, test_data, hidden_dims=mlp_hidden_dims)
    preds_mlp["astroclip_image"] = raw_preds_mlp * scaler["std"] + scaler["mean"]

    preds_mlp["AstroCLIP"] = preds_mlp.pop("astroclip_image")

    if plot:
        plot_scatter_astroclip_only(preds_mlp, k_test)

    mae_knn = np.mean(np.abs(preds_mlp['AstroCLIP'] - k_test))
    r2_knn = r2_score(k_test, preds_mlp['AstroCLIP'])

    stats = {
        "mae_knn" : mae_knn,
        "r2_knn" : r2_knn,
        "z" : z_test
    }

    return preds_mlp, k_test, stats