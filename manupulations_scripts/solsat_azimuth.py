import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile

def load_metadata():
    """Load metadata from JSON file."""
    with open('metadata.json', 'r') as f:
        return json.load(f)

def calibrate_azimuth(raw_azimuth, scale_factor):
    """Calibrate azimuth values."""
    return raw_azimuth * scale_factor

def get_direction(azimuth):
    """Convert azimuth to cardinal direction."""
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    index = int((azimuth + 22.5) // 45 % 8)
    return directions[index]

def create_azimuth_visualization(azimuth_data, output_file, input_meta):
    """Create circular visualization of azimuth data."""
    # Normalize to [0, 360]
    data_normalized = azimuth_data % 360
    
    # Convert to radians for color mapping
    data_radians = np.radians(data_normalized)
    
    # Create HSV color representation
    hsv = np.zeros((*data_normalized.shape, 3))
    hsv[:, :, 0] = data_radians / (2 * np.pi)  # Hue from azimuth
    hsv[:, :, 1] = 1.0  # Full saturation
    hsv[:, :, 2] = 1.0  # Full value
    
    # Convert to RGB
    rgb = plt.cm.hsv(hsv[:, :, 0])[:, :, :3]
    
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
    
    # Get azimuth values and scale factors
    sat_azimuth = metadata['root_attributes']['Sat_Azimuth(Degrees)']
    sun_azimuth = metadata['root_attributes']['Sun_Azimuth(Degrees)']
    
    # Example scale factors (adjust based on actual metadata)
    sat_scale_factor = metadata.get('root_attributes', {}).get('Sat_Azimuth_scale_factor', 1.0)
    sun_scale_factor = metadata.get('root_attributes', {}).get('Sun_Azimuth_scale_factor', 1.0)
    
    # Calibrate azimuths
    cal_sat_azimuth = calibrate_azimuth(sat_azimuth, sat_scale_factor)
    cal_sun_azimuth = calibrate_azimuth(sun_azimuth, sun_scale_factor)
    
    # Create dummy spatial data for visualization
    height, width = 500, 500
    sat_azimuth_grid = np.full((height, width), cal_sat_azimuth)
    sun_azimuth_grid = np.full((height, width), cal_sun_azimuth)
    
    output_files = []
    
    # Save calibrated azimuth data
    for name, data in [("satellite", sat_azimuth_grid), ("solar", sun_azimuth_grid)]:
        # Save raw azimuth data
        azimuth_tiff = f"{name}_azimuth.tif"
        with rasterio.open(azimuth_tiff, 'w',
                          driver='GTiff',
                          height=height,
                          width=width,
                          count=1,
                          dtype=np.float32,
                          crs='EPSG:4326') as dst:
            dst.write(data.astype(np.float32), 1)
        output_files.append(azimuth_tiff)
        
        # Create visualization
        vis_tiff = f"{name}_azimuth_vis.tif"
        create_azimuth_visualization(data, vis_tiff, {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "crs": "EPSG:4326"
        })
        output_files.append(vis_tiff)
    
    # Calculate statistics and directions
    stats = {
        "satellite_azimuth": {
            "raw": float(sat_azimuth),
            "calibrated": float(cal_sat_azimuth),
            "direction": get_direction(cal_sat_azimuth)
        },
        "solar_azimuth": {
            "raw": float(sun_azimuth),
            "calibrated": float(cal_sun_azimuth),
            "direction": get_direction(cal_sun_azimuth)
        },
        "scale_factors": {
            "satellite": float(sat_scale_factor),
            "solar": float(sun_scale_factor)
        }
    }
    
    stats_file = "azimuth_calibration.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    output_files.append(stats_file)
    
    # Zip results
    zip_filename = "azimuth_calibration_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_files:
            zipf.write(file, os.path.basename(file))
    
    print(f"Azimuth calibration completed! Results saved in {zip_filename}")
    print("\nCalibrated Azimuths:")
    print(f"Satellite: {cal_sat_azimuth:.2f}° ({stats['satellite_azimuth']['direction']})")
    print(f"Solar: {cal_sun_azimuth:.2f}° ({stats['solar_azimuth']['direction']})")

if __name__ == "__main__":
    main()