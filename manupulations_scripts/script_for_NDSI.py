import json
import rasterio
import h5py
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
import os
import zipfile

def load_metadata():
    """Load metadata from JSON file."""
    with open('metadata.json', 'r') as f:
        return json.load(f)

def process_band_for_ndsi(h5_file, band_name, metadata):
    """Process band and return scaled radiance."""
    with h5py.File(h5_file, 'r') as f:
        data = f[band_name][:]
        data = np.squeeze(data)
        
        band_attrs = metadata['datasets'][band_name]['attributes']
        scale_factor = band_attrs['lab_radiance_scale_factor']
        offset = band_attrs['lab_radiance_add_offset']
        
        scaled_data = data.astype(float) * scale_factor + offset
        return scaled_data

def calculate_ndsi(green_band, swir_band):
    """Calculate NDSI."""
    # Avoid division by zero
    denominator = green_band + swir_band
    ndsi = np.where(denominator != 0,
                    (green_band - swir_band) / denominator,
                    0)
    # Clip values to [-1, 1] range
    return np.clip(ndsi, -1, 1)

def apply_jet_colormap(data, output_file, input_meta):
    """Apply jet colormap to NDSI data."""
    # Normalize from [-1,1] to [0,1] for visualization
    data_normalized = (data + 1) / 2
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
    
    # Process VIS and SWIR bands
    green_data = process_band_for_ndsi(h5_file, 'IMG_VIS', metadata)
    swir_data = process_band_for_ndsi(h5_file, 'IMG_SWIR', metadata)
    
    # Calculate NDSI
    ndsi = calculate_ndsi(green_data, swir_data)
    
    # Calculate transform
    transform = rasterio.transform.from_bounds(
        left_lon, lower_lat, right_lon, upper_lat,
        ndsi.shape[1], ndsi.shape[0]
    )
    
    output_files = []
    
    # Save NDSI as TIFF
    ndsi_tiff = "ndsi_result.tif"
    with rasterio.open(ndsi_tiff, 'w',
                      driver='GTiff',
                      height=ndsi.shape[0],
                      width=ndsi.shape[1],
                      count=1,
                      dtype=np.float32,
                      crs='EPSG:4326',
                      transform=transform) as dst:
        dst.write(ndsi.astype(np.float32), 1)
    output_files.append(ndsi_tiff)
    
    # Create colored version
    ndsi_colored = "ndsi_result_colored.tif"
    apply_jet_colormap(ndsi, ndsi_colored, {
        "driver": "GTiff",
        "height": ndsi.shape[0],
        "width": ndsi.shape[1],
        "transform": transform,
        "crs": "EPSG:4326"
    })
    output_files.append(ndsi_colored)
    
    # Calculate snow cover statistics
    snow_threshold = 0.4  # Typical threshold for snow
    snow_pixels = np.sum(ndsi > snow_threshold)
    total_pixels = ndsi.size
    snow_coverage = (snow_pixels / total_pixels) * 100
    
    # Save statistics
    stats = {
        "min_ndsi": float(ndsi.min()),
        "max_ndsi": float(ndsi.max()),
        "mean_ndsi": float(ndsi.mean()),
        "std_ndsi": float(ndsi.std()),
        "snow_coverage_percent": float(snow_coverage),
        "snow_threshold_used": snow_threshold
    }
    
    stats_file = "ndsi_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    output_files.append(stats_file)
    
    # Zip results
    zip_filename = "ndsi_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_files:
            zipf.write(file, os.path.basename(file))
    
    print(f"NDSI processing completed! Results saved in {zip_filename}")
    print(f"NDSI range: {stats['min_ndsi']:.3f} to {stats['max_ndsi']:.3f}")
    print(f"Snow coverage: {snow_coverage:.1f}%")

if __name__ == "__main__":
    main()