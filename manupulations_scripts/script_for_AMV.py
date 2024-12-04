import json
import rasterio
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile

def load_metadata():
    """Load metadata from JSON file."""
    with open('metadata.json', 'r') as f:
        return json.load(f)

def process_band_for_amv(h5_file, band_name, metadata):
    """Process a band and return scaled radiance."""
    with h5py.File(h5_file, 'r') as f:
        data = f[band_name][:]
        data = np.squeeze(data)
        
        band_attrs = metadata['datasets'][band_name]['attributes']
        scale_factor = band_attrs['lab_radiance_scale_factor']
        offset = band_attrs['lab_radiance_add_offset']
        
        scaled_data = data.astype(float) * scale_factor + offset
        return scaled_data

def calculate_amv(mir_data, wv_data):
    """Calculate Atmospheric Motion Vectors."""
    return mir_data - wv_data

def apply_jet_colormap(data, output_file, input_meta):
    """Apply jet colormap to AMV data."""
    data_normalized = (data - data.min()) / (data.max() - data.min())
    cmap = plt.get_cmap('jet')
    colored_data = cmap(data_normalized)[:, :, :3]
    
    out_meta = input_meta.copy()
    out_meta.update({
        "count": 3,
        "dtype": "float32"
    })
    
    with rasterio.open(output_file, "w", **out_meta) as dest:
        for i in range(3):
            dest.write(colored_data[:,:,i].astype(np.float32), i+1)
    return output_file

def main():
    # Load metadata
    metadata = load_metadata()
    h5_file = "3RIMG_04SEP2024_1015_L1C_ASIA_MER_V01R00.h5"
    
    # Get spatial reference info
    left_lon = metadata['root_attributes']['left_longitude']
    right_lon = metadata['root_attributes']['right_longitude']
    lower_lat = metadata['root_attributes']['lower_latitude']
    upper_lat = metadata['root_attributes']['upper_latitude']
    
    # Process MIR and WV bands
    mir_data = process_band_for_amv(h5_file, 'IMG_MIR', metadata)
    wv_data = process_band_for_amv(h5_file, 'IMG_WV', metadata)
    
    # Calculate AMV
    amv = calculate_amv(mir_data, wv_data)
    
    # Calculate transform
    transform = rasterio.transform.from_bounds(
        left_lon, lower_lat, right_lon, upper_lat,
        amv.shape[1], amv.shape[0]
    )
    
    output_files = []
    
    # Save AMV as TIFF
    amv_tiff = "amv_result.tif"
    with rasterio.open(amv_tiff, 'w',
                      driver='GTiff',
                      height=amv.shape[0],
                      width=amv.shape[1],
                      count=1,
                      dtype=np.float32,
                      crs='EPSG:4326',
                      transform=transform) as dst:
        dst.write(amv.astype(np.float32), 1)
    output_files.append(amv_tiff)
    
    # Create colored version
    amv_colored = "amv_result_colored.tif"
    apply_jet_colormap(amv, amv_colored, {
        "driver": "GTiff",
        "height": amv.shape[0],
        "width": amv.shape[1],
        "transform": transform,
        "crs": "EPSG:4326"
    })
    output_files.append(amv_colored)
    
    # Calculate and save statistics
    stats = {
        "min_amv": float(amv.min()),
        "max_amv": float(amv.max()),
        "mean_amv": float(amv.mean()),
        "std_amv": float(amv.std()),
        "units": "radiance_difference"
    }
    
    stats_file = "amv_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    output_files.append(stats_file)
    
    # Zip results
    zip_filename = "amv_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_files:
            zipf.write(file, os.path.basename(file))
    
    print(f"AMV processing completed! Results saved in {zip_filename}")
    print(f"AMV range: {stats['min_amv']:.6f} to {stats['max_amv']:.6f}")

if __name__ == "__main__":
    main()