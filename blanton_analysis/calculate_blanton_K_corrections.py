import os
import argparse
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import kcorrect.kcorrect
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset

def load_astroclip_dataset(dataset_path, split, columns):
    """Load only specific columns of the AstroCLIP dataset."""
    print(f"Streaming AstroCLIP dataset from: {dataset_path} (split: {split})")
    dataset = load_dataset(dataset_path, split=split, streaming=True)
    filtered_data = (({col: sample[col] for col in columns}) for sample in dataset)
    return filtered_data

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
    
    results["kcorrs_sdss_01_g"] = k_dered[0]
    results["kcorrs_sdss_01_r"] = k_dered[1]
    results["kcorrs_sdss_01_z"] = k_dered[2]

    absmag = kc.absmag(redshift=redshift, maggies=dered_maggies, ivar=ivar, coeffs=coeffs)
    results["absmag_g"] = absmag[0]
    results["absmag_r"] = absmag[1]
    results["absmag_z"] = absmag[2]
    results["z"] = redshift

    return results


def process_galaxy(galaxy, photometry, kc, noise_range):
    """Process a single galaxy and compute K-corrections."""
    try:
        noise = random.uniform(-noise_range, noise_range)  # Noise between -noise_range and +noise_range
        noisy_redshift = max(0, galaxy["redshift"] + noise)  # Ensure redshift remains non-negative
        result = {
            galaxy["targetid"]: calculate_k_corrections_for_target(
                galaxy["targetid"], noisy_redshift, photometry, kc
            )
        }
        return result
    except Exception as e:
        print(f"Error processing galaxy {galaxy['targetid']} with redshift {galaxy['redshift']}: {e}")
        return {}

def process_galaxies(df, photometry, kc, noise_range):
    """Process galaxies and compute K-corrections in parallel."""
    blanton_kcorrs = {}
    with ProcessPoolExecutor(max_workers=6) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(process_galaxy, galaxy, photometry, kc, noise_range): galaxy for galaxy in df}
        for future in tqdm(as_completed(futures), total=len(df), desc="Processing galaxies"):
            result = future.result()
            blanton_kcorrs.update(result)  # Merge results from each process
    return blanton_kcorrs

def save_k_corrections(blanton_kcorrs, save_path):
    """Save K-corrections to a pickle file."""
    kcorr_df = pd.DataFrame({
        'targetid': blanton_kcorrs.keys(),
        'kcorr_sdss_01_g': [blanton_kcorrs[k]['kcorrs_sdss_01_g'] for k in blanton_kcorrs],
        'kcorr_sdss_01_r': [blanton_kcorrs[k]['kcorrs_sdss_01_r'] for k in blanton_kcorrs],
        'kcorr_sdss_01_z': [blanton_kcorrs[k]['kcorrs_sdss_01_z'] for k in blanton_kcorrs],
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
    parser.add_argument("--noise_range", type=float, default=0.0, help="Range of noise to apply to redshift (default: 0.0).")
    args = parser.parse_args()

    # Load datasets
    astroclip_train = load_astroclip_dataset(args.astroclip_path, split="train", columns=["targetid", "redshift"])
    astroclip_test = load_astroclip_dataset(args.astroclip_path, split="test", columns=["targetid", "redshift"])
    photometry = load_photometry_data(args.desi_path)

    # Initialize kcorrect
    responses_in = ['bass_g', 'bass_r', 'mzls_z']
    responses_out = ['sdss_g0', 'sdss_r0', 'sdss_z0']
    kc = kcorrect.kcorrect.Kcorrect(responses=responses_in, responses_out=responses_out, abcorrect=False)

    # Process galaxies
    print("Processing train dataset...")
    blanton_kcorrs_train = process_galaxies(list(astroclip_train), photometry, kc, args.noise_range)

    print("Processing test dataset...")
    blanton_kcorrs_test = process_galaxies(list(astroclip_test), photometry, kc, args.noise_range)

    # Combine results and save
    blanton_kcorrs_train.update(blanton_kcorrs_test)
    save_k_corrections(blanton_kcorrs_train, args.save_path)