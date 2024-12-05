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

def process_wv_band(h5_file, metadata, normalization_factor=1.0):
    """Process Water Vapor band and calculate content."""
    with h5py.File(h5_file, 'r') as f:
        data = f['IMG_WV'][:]
        data = np.squeeze(data)
        
        band_attrs = metadata['datasets']['IMG_WV']['attributes']
        scale_factor = band_attrs['lab_radiance_scale_factor']
        offset = band_attrs['lab_radiance_add_offset']
        
        # Scale radiance and calculate water vapor content
        wv_radiance = data.astype(float) * scale_factor + offset
        wv_content = 100 * (wv_radiance / normalization_factor)
        return wv_content

def apply_jet_colormap(data, output_file, input_meta):
    """Apply jet colormap to water vapor content data."""
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
    
    # Calculate water vapor content
    wv_content = process_wv_band(h5_file, metadata)
    
    # Calculate transform
    transform = rasterio.transform.from_bounds(
        left_lon, lower_lat, right_lon, upper_lat,
        wv_content.shape[1], wv_content.shape[0]
    )
    
    output_files = []
    
    # Save water vapor content as TIFF
    wv_tiff = "water_vapor_content.tif"
    with rasterio.open(wv_tiff, 'w',
                      driver='GTiff',
                      height=wv_content.shape[0],
                      width=wv_content.shape[1],
                      count=1,
                      dtype=np.float32,
                      crs='EPSG:4326',
                      transform=transform) as dst:
        dst.write(wv_content.astype(np.float32), 1)
    output_files.append(wv_tiff)
    
    # Create colored version
    wv_colored = "water_vapor_content_colored.tif"
    apply_jet_colormap(wv_content, wv_colored, {
        "driver": "GTiff",
        "height": wv_content.shape[0],
        "width": wv_content.shape[1],
        "transform": transform,
        "crs": "EPSG:4326"
    })
    output_files.append(wv_colored)
    
    # Calculate classifications
    wv_levels = {
        "very_dry": (0, 20),
        "dry": (20, 40),
        "moderate": (40, 60),
        "humid": (60, 80),
        "very_humid": (80, float('inf'))
    }
    
    # Calculate statistics
    stats = {
        "min_wv": float(wv_content.min()),
        "max_wv": float(wv_content.max()),
        "mean_wv": float(wv_content.mean()),
        "std_wv": float(wv_content.std()),
        "classifications": {}
    }
    
    # Calculate percentage for each humidity level
    total_pixels = wv_content.size
    for level, (min_val, max_val) in wv_levels.items():
        pixels_in_range = np.sum((wv_content >= min_val) & (wv_content < max_val))
        percentage = (pixels_in_range / total_pixels) * 100
        stats["classifications"][level] = {
            "pixel_count": int(pixels_in_range),
            "percentage": float(percentage),
            "range": f"{min_val}-{max_val}%"
        }
    
    stats_file = "water_vapor_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    output_files.append(stats_file)
    
    # Zip results
    zip_filename = "water_vapor_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_files:
            zipf.write(file, os.path.basename(file))
    
    print(f"Water vapor content processing completed! Results saved in {zip_filename}")
    print(f"Water vapor content range: {stats['min_wv']:.1f}% to {stats['max_wv']:.1f}%")
    print("\nHumidity Classifications:")
    for level, info in stats["classifications"].items():
        print(f"{level.title()}: {info['percentage']:.1f}%")

if __name__ == "__main__":
    main()