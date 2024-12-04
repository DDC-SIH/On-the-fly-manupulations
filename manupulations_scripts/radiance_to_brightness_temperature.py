import json
import rasterio
import h5py
import numpy as np
import matplotlib.pyplot as plt
from rasterio.mask import mask
from pyproj import Transformer
from shapely.geometry import box, mapping
import os
import zipfile

def load_metadata():
    """Load metadata from JSON file."""
    with open('metadata.json', 'r') as f:
        return json.load(f)

def transform_geometry_to_crs(geometry, src_crs="EPSG:4326", dst_crs="EPSG:3857"):
    """Transform coordinates to target CRS."""
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    transformed_coords = [
        [transformer.transform(x, y) for x, y in ring]
        for ring in geometry['coordinates']
    ]
    return {"type": geometry['type'], "coordinates": transformed_coords}

def radiance_to_brightness(data, scale_factor, offset):
    """Convert radiance to brightness temperature."""
    return (data.astype(float) * scale_factor + offset) - 273.15

def process_band(h5_file, band_name, metadata):
    """Process a specific band from H5 file."""
    with h5py.File(h5_file, 'r') as f:
        data = f[band_name][:]
        data = np.squeeze(data)  # Remove single dimensions
        
        band_attrs = metadata['datasets'][band_name]['attributes']
        scale_factor = band_attrs['lab_radiance_scale_factor']
        offset = band_attrs['lab_radiance_add_offset']
        
        brightness = radiance_to_brightness(data, scale_factor, offset)
        return brightness

def apply_jet_colormap(data, output_file, input_meta):
    """Apply jet colormap to data and save as TIFF."""
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

def crop_tiff(input_tiff, output_tiff, geojson_geometry):
    """Crop TIFF using geometry."""
    with rasterio.open(input_tiff) as src:
        geometry = transform_geometry_to_crs(geojson_geometry, 
                                          src_crs="EPSG:4326", 
                                          dst_crs=src.crs.to_string())
        out_image, out_transform = mask(src, [geometry], crop=True)
        
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        with rasterio.open(output_tiff, "w", **out_meta) as dest:
            dest.write(out_image)

def zip_results(files_to_zip, output_zip):
    """Zip output files."""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            zipf.write(file, os.path.basename(file))

def main():
    # Load metadata
    metadata = load_metadata()
    h5_file = "3RIMG_04SEP2024_1015_L1C_ASIA_MER_V01R00.h5"
    
    # Get spatial reference info
    left_lon = metadata['root_attributes']['left_longitude']
    right_lon = metadata['root_attributes']['right_longitude']
    lower_lat = metadata['root_attributes']['lower_latitude']
    upper_lat = metadata['root_attributes']['upper_latitude']
    
    # Process bands
    bands_to_process = ['IMG_TIR1', 'IMG_TIR2']
    output_files = []
    
    for band in bands_to_process:
        # Convert radiance to brightness
        brightness_data = process_band(h5_file, band, metadata)
        
        # Calculate transform
        transform = rasterio.transform.from_bounds(
            left_lon, lower_lat, right_lon, upper_lat,
            brightness_data.shape[1], brightness_data.shape[0]
        )
        
        # Save brightness temperature TIFF
        output_tiff = f"{band}_brightness.tif"
        with rasterio.open(output_tiff, 'w',
                          driver='GTiff',
                          height=brightness_data.shape[0],
                          width=brightness_data.shape[1],
                          count=1,
                          dtype=brightness_data.dtype,
                          crs='EPSG:4326',
                          transform=transform) as dst:
            dst.write(brightness_data, 1)
        output_files.append(output_tiff)
        
        # Create colored version
        colored_tiff = f"{band}_brightness_colored.tif"
        apply_jet_colormap(brightness_data, colored_tiff, dst.meta)
        output_files.append(colored_tiff)
    
    # Zip results
    zip_filename = "brightness_results.zip"
    zip_results(output_files, zip_filename)
    print(f"Processing completed! Results saved in {zip_filename}")

if __name__ == "__main__":
    main()