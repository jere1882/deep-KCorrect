import numpy as np
import h5py
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
import yaml
import os

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

def convert_hdf5_to_pandas(hdf5_path):
    # Define the columns to be selected
    selected_columns = [
        "TARGETID", "Z", "KCORR01_SDSS_R", "KCORR01_SDSS_G", "KCORR01_SDSS_Z",
        "image_embeddings", "ABSMAG01_SDSS_R", "ABSMAG01_SDSS_G", "ABSMAG01_SDSS_Z"
    ]

    # Load the data using the provided function
    data_dict = load_selected_columns_from_hdf5_chunks(hdf5_path, selected_columns)

    # Convert scalar columns to a DataFrame
    df = pd.DataFrame({key: data_dict[key] for key in data_dict if key != 'image_embeddings'})

    # Convert image_embeddings into 1024 separate columns
    image_embeddings_df = pd.DataFrame(data_dict['image_embeddings'], columns=[f"image_emb_{i}" for i in range(1024)])

    # Concatenate the two DataFrames
    df = pd.concat([df, image_embeddings_df], axis=1)

    # Create the output filename
    base, ext = os.path.splitext(hdf5_path)
    output_filename = f"{base}_pd.h5"

    # Save the DataFrame to a CSV file
    df.to_hdf(output_filename, key="data", mode="w")
    print(f"Data saved to {output_filename}")

def get_dataset_slice_between_redshifts(dataset, z_min, z_max):
    z = dataset['Z']
    cond = (z >= z_min) & (z <= z_max)
    return dataset[cond]

def load_dataset_between_redshifts(path, z_min, z_max):
    dataset = pd.read_hdf(path, key='data')
    return get_dataset_slice_between_redshifts(dataset, z_min, z_max)

def prepare_dataset(train_ds, test_ds, z_min, z_max, target_variable, include_redshift=False):
    # Retrieve the embeddings
    X_train = train_ds.filter(like="image_emb_")
    X_test = test_ds.filter(like="image_emb_")
    
    if include_redshift:
        # Include the redshift column if specified
        X_train = pd.concat([X_train, train_ds[['Z']]], axis=1)
        X_test = pd.concat([X_test, test_ds[['Z']]], axis=1)

    # Scale the data
    embedding_scaler = StandardScaler().fit(X_train)
    X_train = embedding_scaler.transform(X_train)
    X_test = embedding_scaler.transform(X_test)

    # Retrieve the redshift vectors
    z_train = train_ds['Z']
    z_test = test_ds['Z']

    cond_train = (z_train >= z_min) & (z_train <= z_max)
    cond_test = (z_test >= z_min) & (z_test <= z_max)
        
    X_train = X_train[cond_train]
    X_test = X_test[cond_test]

    y_train = train_ds[target_variable][cond_train]
    y_test = test_ds[target_variable][cond_test]

    return X_train, y_train, X_test, y_test

def prepare_dataset_for_visualization(train_ds, test_ds, z_min, z_max, target_variable, include_redshift=False):
    # Retrieve the embeddings
    X_train = train_ds.filter(like="image_emb_")
    X_test = test_ds.filter(like="image_emb_")
    
    if include_redshift:
        # Include the redshift column if specified
        X_train = pd.concat([X_train, train_ds[['Z']]], axis=1)
        X_test = pd.concat([X_test, test_ds[['Z']]], axis=1)

    # Scale the data
    embedding_scaler = StandardScaler().fit(X_train)
    X_train = embedding_scaler.transform(X_train)
    X_test = embedding_scaler.transform(X_test)

    # Retrieve the redshift vectors
    z_train = train_ds['Z']
    z_test = test_ds['Z']

    cond_train = (z_train >= z_min) & (z_train <= z_max)
    cond_test = (z_test >= z_min) & (z_test <= z_max)
        
    X_train = X_train[cond_train]
    X_test = X_test[cond_test]

    y_train = train_ds[target_variable][cond_train]
    y_test = test_ds[target_variable][cond_test]

    return X_train, y_train, X_test, y_test, z_train[cond_train], z_test[cond_test]

def load_data(train_path, test_path, z_min, z_max, target_variable, include_redshift=False):
    logging.debug("Loading training data from: %s", train_path)
    train_ds = pd.read_hdf(train_path, key='data')
    logging.debug("Training data shape: %s", train_ds.shape)

    logging.debug("Loading test data from: %s", test_path)
    test_ds = pd.read_hdf(test_path, key='data')
    logging.debug("Test data shape: %s", test_ds.shape)

    X_train, y_train, X_test, y_test = prepare_dataset(
        train_ds, test_ds, z_min, z_max, target_variable, include_redshift
    )
    logging.debug("Data prepared. X_train shape: %s, y_train shape: %s", X_train.shape, y_train.shape)

    # Debugging statements
    logging.debug("Loading data for target_variable: %s, z_min: %f, z_max: %f", target_variable, z_min, z_max)
    logging.debug("X_train shape: %s, y_train type: %s", X_train.shape, type(y_train))
    logging.debug("X_test shape: %s, y_test type: %s", X_test.shape, type(y_test))
    
    # Check if y_train or y_test is empty
    if y_train.empty or y_test.empty:
        logging.warning("Empty data for target_variable: %s, z_min: %f, z_max: %f", target_variable, z_min, z_max)

    return X_train, y_train, X_test, y_test

def load_config(config_path):
    logging.debug("Loading model configuration from: %s", config_path)
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)  # Use yaml.safe_load to parse the YAML file
    logging.debug("Model configuration loaded: %s", config)
    return config

def collect_predictions(test_path="data/test_df_kcorr_panda.h5", 
                       tensorboard_log_dir="tensorboard_logs/AstroCLIP",
                       experiment_name=None):
    
    test_ds = pd.read_hdf(test_path, key='data')
    image_emb_cols = [col for col in test_ds.columns if col.startswith('image_emb')]
    test_ds = test_ds.drop(columns=image_emb_cols)

    # We will iterate over the folders in the tensorboard log directory
    # and extend the test dataset with the predictions
    for folder in os.listdir(tensorboard_log_dir):
        # Check if the folder is a valid model checkpoint
        if os.path.isdir(os.path.join(tensorboard_log_dir, folder)):
            # Identify the config file
            config_file = os.path.join(tensorboard_log_dir, folder, 'config.txt')
            if os.path.exists(config_file):
                # Load the config
                config = load_config(config_file)

                if experiment_name and experiment_name != config.get('logging', {}).get('experiment_tag', ''):
                    continue
                    
                # Rest of the processing remains the same
                z_min = config['data']['z_min']
                z_max = config['data']['z_max']
                cond = (test_ds['Z'] >= z_min) & (test_ds['Z'] <= z_max)
                column_name = f"{config['data']['target_variable']}_{config['data']['z_min']}_{config['data']['z_max']}_{config['model']['model_type']}_{config['model']['n_hidden']}_{config['data']['include_redshift']}"
                
                # Check if the folder contains a 'last_epoch_predictions.npy' file
                predictions_file = os.path.join(tensorboard_log_dir, folder, 'last_epoch_predictions.npy')
                if os.path.exists(predictions_file):
                    # Load the predictions
                    predictions = np.load(predictions_file)
                    # Extend the test dataset with the predictions, and use a void value for the rows that are not in the subset
                    test_ds.loc[cond, column_name] = predictions
                    test_ds.loc[~cond, column_name] = np.nan
                
    return test_ds