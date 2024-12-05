import os
import re
import time
from scripttogeneratejson import (
    extract_h5_metadata, 
    save_json_metadata, 
    process_tiff_metadata
)
import subprocess
import glob

def determine_product_level(filename):
    """Determine if file is L1B, L1C, or L2C based on filename."""
    if 'L1C' in filename:
        return 'L1C'
    elif 'L1B' in filename:
        return 'L1B'
    elif 'L2C' in filename:
        return 'L2C'
    else:
        raise ValueError("Unknown product level in filename")

def wait_for_tiff_files(bands, timeout=60):
    """Wait for TIFF files to be generated and collect their paths."""
    start_time = time.time()
    tiff_files = set()
    
    while time.time() - start_time < timeout:
        for band_name in bands:
            pattern = f"IMG_{band_name}*.tif"
            found_files = glob.glob(pattern)
            tiff_files.update(found_files)
            
        if len(tiff_files) >= len(bands):
            break
        time.sleep(1)
    
    return list(tiff_files)

def process_satellite_data(input_h5_file):
    """Main workflow function to process satellite data."""
    # Step 1: Generate H5 metadata
    print("Generating H5 metadata...")
    h5_metadata = extract_h5_metadata(input_h5_file)
    metadata_filename = f"{os.path.splitext(input_h5_file)[0]}_metadata.json"
    save_json_metadata(h5_metadata, metadata_filename)
    print(f"Saved H5 metadata to {metadata_filename}")

    # Define bands to process
    bands = ["VIS", "MIR", "SWIR", "TIR1", "TIR2", "WV"]

    # Step 2: Determine product level and process accordingly
    product_level = determine_product_level(input_h5_file)
    print(f"Detected product level: {product_level}")

    # Step 3: Process based on product level
    try:
        if product_level == 'L1C':
            print("Processing L1C data...")
            subprocess.run(['python', 'l1c.py'], check=True)
        elif product_level == 'L1B':
            print("Processing L1B data...")
            subprocess.run(['python', 'l1b.py'], check=True)
        elif product_level == 'L2C':
            print("Processing L2C data...")
            print("L2C processing not implemented yet")
            return
        
        # Step 4: Wait for TIFF files and generate metadata
        print("Waiting for TIFF files to be generated...")
        tiff_files = wait_for_tiff_files(bands)
        
        if tiff_files:
            print("Generating metadata for TIFF files...")
            for tiff_file in tiff_files:
                process_tiff_metadata(tiff_file)
            print(f"Generated metadata for {len(tiff_files)} TIFF files")
        else:
            print("No TIFF files found within timeout period")
        
        print(f"Successfully processed {product_level} data")
        
    except subprocess.CalledProcessError as e:
        print(f"Error processing {product_level} data: {str(e)}")
        raise

def main():
    input_file = "3RIMG_04SEP2024_1015_L1C_ASIA_MER_V01R00.h5"
    
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
            
        process_satellite_data(input_file)
        print("Workflow completed successfully")
        
    except Exception as e:
        print(f"Error in workflow: {str(e)}")
        raise

if __name__ == "__main__":
    main()