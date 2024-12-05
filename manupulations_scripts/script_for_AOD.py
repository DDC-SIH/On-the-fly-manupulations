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

def process_band_for_aod(h5_file, metadata, epsilon=0.1):
    """Process VIS band for AOD calculation."""
    with h5py.File(h5_file, 'r') as f:
        data = f['IMG_VIS'][:]
        data = np.squeeze(data)
        
        band_attrs = metadata['datasets']['IMG_VIS']['attributes']
        scale_factor = band_attrs['lab_radiance_scale_factor']
        offset = band_attrs['lab_radiance_add_offset']
        
        # Scale radiance
        vis_radiance = data.astype(float) * scale_factor + offset
        
        # Calculate AOD
        aod = vis_radiance / (vis_radiance + epsilon)
        return aod

def apply_jet_colormap(data, output_file, input_meta):
    """Apply jet colormap to AOD data."""
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
    
    # Calculate AOD
    aod = process_band_for_aod(h5_file, metadata)
    
    # Calculate transform
    transform = rasterio.transform.from_bounds(
        left_lon, lower_lat, right_lon, upper_lat,
        aod.shape[1], aod.shape[0]
    )
    
    output_files = []
    
    # Save AOD as TIFF
    aod_tiff = "aod_result.tif"
    with rasterio.open(aod_tiff, 'w',
                      driver='GTiff',
                      height=aod.shape[0],
                      width=aod.shape[1],
                      count=1,
                      dtype=np.float32,
                      crs='EPSG:4326',
                      transform=transform) as dst:
        dst.write(aod.astype(np.float32), 1)
    output_files.append(aod_tiff)
    
    # Create colored version
    aod_colored = "aod_result_colored.tif"
    apply_jet_colormap(aod, aod_colored, {
        "driver": "GTiff",
        "height": aod.shape[0],
        "width": aod.shape[1],
        "transform": transform,
        "crs": "EPSG:4326"
    })
    output_files.append(aod_colored)
    
    # Calculate AOD statistics
    # Define AOD thresholds
    aod_levels = {
        "clear": (0.0, 0.1),
        "moderate": (0.1, 0.3),
        "hazy": (0.3, 0.5),
        "very_hazy": (0.5, float('inf'))
    }
    
    stats = {
        "min_aod": float(aod.min()),
        "max_aod": float(aod.max()),
        "mean_aod": float(aod.mean()),
        "std_aod": float(aod.std()),
        "epsilon_used": 0.1,
        "aod_classification": {}
    }
    
    # Calculate percentage for each AOD level
    total_pixels = aod.size
    for level, (min_val, max_val) in aod_levels.items():
        pixels_in_range = np.sum((aod >= min_val) & (aod < max_val))
        percentage = (pixels_in_range / total_pixels) * 100
        stats["aod_classification"][level] = {
            "pixel_count": int(pixels_in_range),
            "percentage": float(percentage)
        }
    
    stats_file = "aod_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    output_files.append(stats_file)
    
    # Zip results
    zip_filename = "aod_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_files:
            zipf.write(file, os.path.basename(file))
    
    print(f"AOD processing completed! Results saved in {zip_filename}")
    print(f"AOD range: {stats['min_aod']:.3f} to {stats['max_aod']:.3f}")
    print("\nAOD Classification:")
    for level, info in stats["aod_classification"].items():
        print(f"{level.title()}: {info['percentage']:.1f}%")

if __name__ == "__main__":
    main()