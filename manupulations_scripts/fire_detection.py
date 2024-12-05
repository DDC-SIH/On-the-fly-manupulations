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

def radiance_to_brightness_kelvin(data, scale_factor, offset):
    """Convert radiance to brightness temperature in Kelvin."""
    return (data.astype(float) * scale_factor + offset)

def detect_fires(temperature_data, threshold=350):
    """Create fire mask based on temperature threshold."""
    return (temperature_data > threshold).astype(np.uint8)

def process_band_for_fires(h5_file, band_name, metadata):
    """Process TIR band for fire detection."""
    with h5py.File(h5_file, 'r') as f:
        data = f[band_name][:]
        data = np.squeeze(data)
        
        band_attrs = metadata['datasets'][band_name]['attributes']
        scale_factor = band_attrs['lab_radiance_scale_factor']
        offset = band_attrs['lab_radiance_add_offset']
        
        brightness = radiance_to_brightness_kelvin(data, scale_factor, offset)
        return brightness

def create_fire_visualization(fire_mask, temperature_data, output_file, input_meta):
    """Create RGB visualization: Red for fires, grayscale for temperature."""
    # Normalize temperature for background
    temp_normalized = (temperature_data - temperature_data.min()) / (temperature_data.max() - temperature_data.min())
    
    # Create RGB image
    rgb = np.zeros((temperature_data.shape[0], temperature_data.shape[1], 3))
    rgb[:, :, 0] = np.where(fire_mask == 1, 1.0, temp_normalized)  # Red for fires
    rgb[:, :, 1] = np.where(fire_mask == 1, 0.0, temp_normalized)  # Temperature background
    rgb[:, :, 2] = np.where(fire_mask == 1, 0.0, temp_normalized)  # Temperature background
    
    out_meta = input_meta.copy()
    out_meta.update({
        "count": 3,
        "dtype": "float32"
    })
    
    with rasterio.open(output_file, "w", **out_meta) as dest:
        for i in range(3):
            dest.write(rgb[:,:,i].astype(np.float32), i+1)
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
    
    output_files = []
    
    # Process TIR1 band for fire detection
    temperature = process_band_for_fires(h5_file, 'IMG_TIR1', metadata)
    fire_mask = detect_fires(temperature)
    
    # Calculate transform
    transform = rasterio.transform.from_bounds(
        left_lon, lower_lat, right_lon, upper_lat,
        fire_mask.shape[1], fire_mask.shape[0]
    )
    
    # Save fire mask
    mask_tiff = "fire_mask.tif"
    with rasterio.open(mask_tiff, 'w',
                      driver='GTiff',
                      height=fire_mask.shape[0],
                      width=fire_mask.shape[1],
                      count=1,
                      dtype=np.uint8,
                      crs='EPSG:4326',
                      transform=transform) as dst:
        dst.write(fire_mask, 1)
    output_files.append(mask_tiff)
    
    # Create visualization
    vis_tiff = "fire_detection_vis.tif"
    create_fire_visualization(fire_mask, temperature, vis_tiff, {
        "driver": "GTiff",
        "height": fire_mask.shape[0],
        "width": fire_mask.shape[1],
        "transform": transform,
        "crs": "EPSG:4326"
    })
    output_files.append(vis_tiff)
    
    # Calculate statistics
    fire_pixels = np.sum(fire_mask)
    total_pixels = fire_mask.size
    fire_percentage = (fire_pixels / total_pixels) * 100
    
    max_temp = temperature[fire_mask == 1].max() if fire_pixels > 0 else None
    
    stats = {
        "fire_pixels_count": int(fire_pixels),
        "fire_coverage_percent": float(fire_percentage),
        "max_temperature_k": float(max_temp) if max_temp is not None else None,
        "threshold_used": 350,
        "total_pixels": int(total_pixels)
    }
    
    stats_file = "fire_detection_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    output_files.append(stats_file)
    
    # Zip results
    zip_filename = "fire_detection_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_files:
            zipf.write(file, os.path.basename(file))
    
    print(f"Fire detection completed! Results saved in {zip_filename}")
    print(f"Found {fire_pixels} fire pixels ({fire_percentage:.2f}% coverage)")
    if max_temp is not None:
        print(f"Maximum temperature in fire pixels: {max_temp:.1f} K")

if __name__ == "__main__":
    main()