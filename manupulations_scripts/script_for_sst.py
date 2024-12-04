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

def calculate_sst(radiance_data, scale_factor, offset):
    """Calculate Sea Surface Temperature."""
    # SST = (R * S) + O - 273.15 (convert to Celsius)
    return (radiance_data.astype(float) * scale_factor + offset) - 273.15

def process_band_for_sst(h5_file, metadata):
    """Process TIR2 band for SST calculation."""
    with h5py.File(h5_file, 'r') as f:
        # Extract TIR2 data
        data = f['IMG_TIR2'][:]
        data = np.squeeze(data)
        
        # Get conversion parameters
        band_attrs = metadata['datasets']['IMG_TIR2']['attributes']
        scale_factor = band_attrs['lab_radiance_scale_factor']
        offset = band_attrs['lab_radiance_add_offset']
        
        # Calculate SST
        sst = calculate_sst(data, scale_factor, offset)
        return sst

def apply_jet_colormap(data, output_file, input_meta):
    """Apply jet colormap to SST data."""
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
    
    # Calculate SST
    sst = process_band_for_sst(h5_file, metadata)
    
    # Calculate transform
    transform = rasterio.transform.from_bounds(
        left_lon, lower_lat, right_lon, upper_lat,
        sst.shape[1], sst.shape[0]
    )
    
    output_files = []
    
    # Save SST as TIFF
    sst_tiff = "sst_result.tif"
    with rasterio.open(sst_tiff, 'w',
                      driver='GTiff',
                      height=sst.shape[0],
                      width=sst.shape[1],
                      count=1,
                      dtype=np.float32,
                      crs='EPSG:4326',
                      transform=transform) as dst:
        dst.write(sst.astype(np.float32), 1)
    output_files.append(sst_tiff)
    
    # Create colored version
    sst_colored = "sst_result_colored.tif"
    apply_jet_colormap(sst, sst_colored, {
        "driver": "GTiff",
        "height": sst.shape[0],
        "width": sst.shape[1],
        "transform": transform,
        "crs": "EPSG:4326"
    })
    output_files.append(sst_colored)
    
    # Save statistics
    stats = {
        "min_sst": float(sst.min()),
        "max_sst": float(sst.max()),
        "mean_sst": float(sst.mean()),
        "std_sst": float(sst.std())
    }
    
    stats_file = "sst_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    output_files.append(stats_file)
    
    # Zip results
    zip_filename = "sst_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_files:
            zipf.write(file, os.path.basename(file))
    
    print(f"SST processing completed! Results saved in {zip_filename}")
    print(f"SST range: {stats['min_sst']:.2f}°C to {stats['max_sst']:.2f}°C")

if __name__ == "__main__":
    main()