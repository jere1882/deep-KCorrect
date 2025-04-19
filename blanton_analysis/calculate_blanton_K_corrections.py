import os
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import kcorrect.kcorrect
import pickle

def load_astroclip_dataset(dataset_path, split):
    """Load the specified split of the AstroCLIP dataset."""
    print(f"Loading AstroCLIP dataset from: {dataset_path} (split: {split})")
    return load_dataset(dataset_path, split=split)

def load_photometry_data(file_path, key='df'):
    """Load photometry data from an HDF5 file."""
    print(f"Loading photometry data from: {file_path}")
    return pd.read_hdf(file_path, key=key)

def calculate_k_corrections_for_target(targetid, redshift, photometry, kc):
    """Calculate K-corrections and absolute magnitudes for a specific target."""
    results = {}
    filtered_df = photometry[photometry['targetid'] == targetid]
    if filtered_df.empty:
        raise ValueError(f"No data found for targetid: {targetid}")

    ivar = [filtered_df['flux_ivar_g'].item() * 10**18,
            filtered_df['flux_ivar_r'].item() * 10**18,
            filtered_df['flux_ivar_z'].item() * 10**18]
    dered_maggies = [filtered_df['dered_flux_g'].item() * 10**(-9),
                     filtered_df['dered_flux_r'].item() * 10**(-9),
                     filtered_df['dered_flux_z'].item() * 10**(-9)]

    coeffs = kc.fit_coeffs(redshift=redshift, maggies=dered_maggies, ivar=ivar)
    k_dered = kc.kcorrect(redshift=redshift, coeffs=coeffs, band_shift=0.1)
    
    results["K_G_G"] = k_dered[0]
    results["K_R_G"] = k_dered[1]
    results["K_Z_G"] = k_dered[2]
    results["K_R_R"] = k_dered[3]
    results["K_Z_R"] = k_dered[4]
    results["K_Z_Z"] = k_dered[5]

    absmag = kc.absmag(redshift=redshift, maggies=dered_maggies, ivar=ivar, coeffs=coeffs, band_shift=0.1)
    results["absmag_g_g"] = absmag[0]
    results["absmag_g_r"] = absmag[1]
    results["absmag_g_z"] = absmag[2]
    results["absmag_r_r"] = absmag[3]
    results["absmag_z_r"] = absmag[4]
    results["absmag_z_z"] = absmag[5]
    results["z"] = redshift

    return results

def process_galaxies(df, photometry, kc, save_path):
    """Process galaxies and compute K-corrections."""
    blanton_kcorrs = {}
    error_log = {
        "Maximum number of iterations reached.": [],
        "Matrix is singular.": [],
        "Redshift out of range for interpolating A matrix!": []
    }

    for galaxy in tqdm(df, desc="Processing galaxies"):
        try:
            truncated_redshift = round(galaxy["redshift"], 3)
            blanton_kcorrs[galaxy["targetid"]] = calculate_k_corrections_for_target(
                galaxy["targetid"], truncated_redshift, photometry, kc
            )
        except Exception as e:
            error_message = str(e)
            if error_message in error_log:
                error_log[error_message].append(galaxy["targetid"])
            else:
                print(f"Unexpected error for galaxy {galaxy['targetid']} with redshift {galaxy['redshift']}: {e}")

    # Save the error log
    error_log_path = save_path.replace('.pickle', '_error_log.pickle')
    with open(error_log_path, 'wb') as f:
        pickle.dump(error_log, f)
    print(f"Error log saved to: {error_log_path}")

    return blanton_kcorrs

def save_k_corrections(blanton_kcorrs, save_path):
    """Save K-corrections to a pickle file."""
    kcorr_df = pd.DataFrame({
        'targetid': blanton_kcorrs.keys(),
        'K_G_G': [blanton_kcorrs[k]['K_G_G'] for k in blanton_kcorrs],
        'K_R_G': [blanton_kcorrs[k]['K_R_G'] for k in blanton_kcorrs],
        'K_Z_G': [blanton_kcorrs[k]['K_Z_G'] for k in blanton_kcorrs],
        'K_R_R': [blanton_kcorrs[k]['K_R_R'] for k in blanton_kcorrs],
        'K_Z_R': [blanton_kcorrs[k]['K_Z_R'] for k in blanton_kcorrs],
        'K_Z_Z': [blanton_kcorrs[k]['K_Z_Z'] for k in blanton_kcorrs],
        'absmag_g_g': [blanton_kcorrs[k]['absmag_g_g'] for k in blanton_kcorrs],
        'absmag_g_r': [blanton_kcorrs[k]['absmag_g_r'] for k in blanton_kcorrs],
        'absmag_g_z': [blanton_kcorrs[k]['absmag_g_z'] for k in blanton_kcorrs],
        'absmag_r_r': [blanton_kcorrs[k]['absmag_r_r'] for k in blanton_kcorrs],
        'absmag_z_r': [blanton_kcorrs[k]['absmag_z_r'] for k in blanton_kcorrs],
        'absmag_z_z': [blanton_kcorrs[k]['absmag_z_z'] for k in blanton_kcorrs],
        'z': [blanton_kcorrs[k]['z'] for k in blanton_kcorrs]
    })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    kcorr_df.to_pickle(save_path)
    print(f"K-corrections saved to: {save_path}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process AstroCLIP and DESI-LS datasets.")
    parser.add_argument("--astroclip_path", type=str, required=True, help="Path to the AstroCLIP dataset.")
    parser.add_argument("--desi_path", type=str, required=True, help="Path to the DESI photometry data (HDF5).")
    parser.add_argument("--save_path", type=str, default="../data/blanton_kcorrs.pickle", help="Path to save the K-corrections (pickle).")
    args = parser.parse_args()

    # Load datasets
    astroclip_train = load_astroclip_dataset(args.astroclip_path, split="train")
    astroclip_test = load_astroclip_dataset(args.astroclip_path, split="test")
    photometry = load_photometry_data(args.desi_path)

    # Initialize kcorrect
    responses_in = ['bass_g', 'bass_r', 'mzls_z']
    responses_out = ['sdss_2010-g', 'sdss_2010-g', 'sdss_2010-g', 'sdss_2010-r', 'sdss_2010-r', 'sdss_2010-z']
    responses_map = ['bass_g', 'bass_r', 'mzls_z', 'bass_r', 'mzls_z', 'mzls_z']
    kc = kcorrect.kcorrect.Kcorrect(responses=responses_in, responses_out=responses_out, responses_map=responses_map, abcorrect=False)
    
    # Process galaxies
    print("Processing train dataset...")
    blanton_kcorrs_train = process_galaxies(astroclip_train, photometry, kc, args.save_path)

    print("Processing test dataset...")
    blanton_kcorrs_test = process_galaxies(astroclip_test, photometry, kc, args.save_path)

    # Combine results and save
    blanton_kcorrs_train.update(blanton_kcorrs_test)
    save_k_corrections(blanton_kcorrs_train, args.save_path)
