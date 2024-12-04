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

def calculate_uth(wv_radiance):
    """Calculate Upper Tropospheric Humidity."""
    # UTH = 100 * (WV / (WV + 1))
    return 100 * (wv_radiance / (wv_radiance + 1))

def process_wv_band(h5_file, metadata):
    """Process Water Vapor band data."""
    with h5py.File(h5_file, 'r') as f:
        data = f['IMG_WV'][:]
        data = np.squeeze(data)
        
        band_attrs = metadata['datasets']['IMG_WV']['attributes']
        scale_factor = band_attrs['lab_radiance_scale_factor']
        offset = band_attrs['lab_radiance_add_offset']
        
        # Apply scale factor and offset to radiance
        wv_processed = data.astype(float) * scale_factor + offset
        return wv_processed

def apply_jet_colormap(data, output_file, input_meta):
    """Apply jet colormap to UTH data."""
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
    
    # Process WV band and calculate UTH
    wv_radiance = process_wv_band(h5_file, metadata)
    uth = calculate_uth(wv_radiance)
    
    # Calculate transform
    transform = rasterio.transform.from_bounds(
        left_lon, lower_lat, right_lon, upper_lat,
        uth.shape[1], uth.shape[0]
    )
    
    output_files = []
    
    # Save UTH as TIFF
    uth_tiff = "uth_result.tif"
    with rasterio.open(uth_tiff, 'w',
                      driver='GTiff',
                      height=uth.shape[0],
                      width=uth.shape[1],
                      count=1,
                      dtype=np.float32,
                      crs='EPSG:4326',
                      transform=transform) as dst:
        dst.write(uth.astype(np.float32), 1)
    output_files.append(uth_tiff)
    
    # Create colored version
    uth_colored = "uth_result_colored.tif"
    apply_jet_colormap(uth, uth_colored, {
        "driver": "GTiff",
        "height": uth.shape[0],
        "width": uth.shape[1],
        "transform": transform,
        "crs": "EPSG:4326"
    })
    output_files.append(uth_colored)
    
    # Save statistics
    stats = {
        "min_uth": float(uth.min()),
        "max_uth": float(uth.max()),
        "mean_uth": float(uth.mean()),
        "std_uth": float(uth.std()),
        "units": "percent"
    }
    
    stats_file = "uth_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    output_files.append(stats_file)
    
    # Zip results
    zip_filename = "uth_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_files:
            zipf.write(file, os.path.basename(file))
    
    print(f"UTH processing completed! Results saved in {zip_filename}")
    print(f"UTH range: {stats['min_uth']:.1f}% to {stats['max_uth']:.1f}%")

if __name__ == "__main__":
    main()