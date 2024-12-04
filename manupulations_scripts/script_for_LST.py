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

def calculate_lst(radiance_data, scale_factor, offset):
    """Calculate Land Surface Temperature in Celsius."""
    return (radiance_data.astype(float) * scale_factor + offset) - 273.15

def process_tir1_for_lst(h5_file, metadata):
    """Process TIR1 band for LST calculation."""
    with h5py.File(h5_file, 'r') as f:
        data = f['IMG_TIR1'][:]
        data = np.squeeze(data)
        
        band_attrs = metadata['datasets']['IMG_TIR1']['attributes']
        scale_factor = band_attrs['lab_radiance_scale_factor']
        offset = band_attrs['lab_radiance_add_offset']
        
        lst = calculate_lst(data, scale_factor, offset)
        return lst

def apply_jet_colormap(data, output_file, input_meta):
    """Apply jet colormap to LST data."""
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
    
    # Calculate LST
    lst = process_tir1_for_lst(h5_file, metadata)
    
    # Calculate transform
    transform = rasterio.transform.from_bounds(
        left_lon, lower_lat, right_lon, upper_lat,
        lst.shape[1], lst.shape[0]
    )
    
    output_files = []
    
    # Save LST as TIFF
    lst_tiff = "lst_result.tif"
    with rasterio.open(lst_tiff, 'w',
                      driver='GTiff',
                      height=lst.shape[0],
                      width=lst.shape[1],
                      count=1,
                      dtype=np.float32,
                      crs='EPSG:4326',
                      transform=transform) as dst:
        dst.write(lst.astype(np.float32), 1)
    output_files.append(lst_tiff)
    
    # Create colored version
    lst_colored = "lst_result_colored.tif"
    apply_jet_colormap(lst, lst_colored, {
        "driver": "GTiff",
        "height": lst.shape[0],
        "width": lst.shape[1],
        "transform": transform,
        "crs": "EPSG:4326"
    })
    output_files.append(lst_colored)
    
    # Save statistics
    stats = {
        "min_lst": float(lst.min()),
        "max_lst": float(lst.max()),
        "mean_lst": float(lst.mean()),
        "std_lst": float(lst.std()),
        "units": "celsius"
    }
    
    stats_file = "lst_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    output_files.append(stats_file)
    
    # Zip results
    zip_filename = "lst_results.zip"
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in output_files:
            zipf.write(file, os.path.basename(file))
    
    print(f"LST processing completed! Results saved in {zip_filename}")
    print(f"Temperature range: {stats['min_lst']:.1f}°C to {stats['max_lst']:.1f}°C")

if __name__ == "__main__":
    main()