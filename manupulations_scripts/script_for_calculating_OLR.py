import json
import rasterio
import h5py
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
from pyproj import Transformer
import os
import zipfile

def load_metadata():
    """Load metadata from JSON file."""
    with open('metadata.json', 'r') as f:
        return json.load(f)

def radiance_to_brightness_kelvin(data, scale_factor, offset):
    """Convert radiance to brightness temperature in Kelvin."""
    return (data.astype(float) * scale_factor + offset)

def calculate_olr(tir1_temp, tir2_temp, empirical_constant=1.1):
    """Calculate OLR using TIR1 and TIR2 brightness temperatures."""
    return empirical_constant * (tir1_temp + tir2_temp)

def process_band_for_olr(h5_file, band_name, metadata):
    """Process a band and return brightness temperature in Kelvin."""
    with h5py.File(h5_file, 'r') as f:
        data = f[band_name][:]
        data = np.squeeze(data)
        
        band_attrs = metadata['datasets'][band_name]['attributes']
        scale_factor = band_attrs['lab_radiance_scale_factor']
        offset = band_attrs['lab_radiance_add_offset']
        
        brightness = radiance_to_brightness_kelvin(data, scale_factor, offset)
        return brightness

def apply_jet_colormap(data, output_file, input_meta):
    """Apply jet colormap to OLR data."""
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
    
    # Process TIR1 and TIR2 bands
    tir1_temp = process_band_for_olr(h5_file, 'IMG_TIR1', metadata)
    tir2_temp = process_band_for_olr(h5_file, 'IMG_TIR2', metadata)
    
    # Calculate OLR
    olr = calculate_olr(tir1_temp, tir2_temp)
    
    # Calculate transform
    transform = rasterio.transform.from_bounds(
        left_lon, lower_lat, right_lon, upper_lat,
        olr.shape[1], olr.shape[0]
    )
    
    output_files = []
    
    # Save OLR as TIFF
    olr_tiff = "olr_result.tif"
    with rasterio.open(olr_tiff, 'w',
                      driver='GTiff',
                      height=olr.shape[0],
                      width=olr.shape[1],
                      count=1,
                      dtype=np.float32,
                      crs='EPSG:4326',
                      transform=transform) as dst:
        dst.write(olr.astype(np.float32), 1)
    output_files.append(olr_tiff)
    
    # Create colored version
    olr_colored = "olr_result_colored.tif"
    apply_jet_colormap(olr, olr_colored, {
        "driver": "GTiff",
        "height": olr.shape[0],
        "width": olr.shape[1],
        "transform": transform,
        "crs": "EPSG:4326"
    })
    output_files.append(olr_colored)
    
    # Save statistics
    stats = {
        "min_olr": float(olr.min()),
        "max_olr": float(olr.max()),
        "mean_olr": float(olr.mean()),
        "std_olr": float(olr.std())
    }
    
    with open("olr_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    output_files.append("olr_statistics.json")
    
    # Zip results
    zip_filename = "olr_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_files:
            zipf.write(file, os.path.basename(file))
    
    print(f"OLR processing completed! Results saved in {zip_filename}")

if __name__ == "__main__":
    main()