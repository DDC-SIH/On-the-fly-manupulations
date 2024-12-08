import json
import rasterio
import numpy as np
from rasterio.mask import mask
import requests
import os
from rasterio.warp import transform_geom
import logging
import zipfile
from rasterio.features import geometry_mask
from matplotlib import cm
from matplotlib.colors import Normalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_tiff(url, local_filename):
    """Download TIFF file from URL."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return local_filename
    else:
        raise Exception(f"Failed to download {url}")

def create_mask(geometry, out_shape, transform):
    """Create a mask from geometry"""
    mask = geometry_mask(
        [geometry],
        out_shape=out_shape,
        transform=transform,
        invert=True
    )
    return mask

def crop_tiff(input_tiff, geometry):
    """Crop a TIFF file based on the geometry."""
    try:
        with rasterio.open(input_tiff) as src:
            # Transform geometry to match raster's CRS
            transformed_geometry = transform_geom(
                'EPSG:4326',  # Source CRS (assuming geometry is in WGS84)
                src.crs,      # Target CRS from the raster
                geometry
            )
            
            logger.info(f"Raster bounds: {src.bounds}")
            logger.info(f"Geometry bounds: {transformed_geometry}")
            
            out_image, out_transform = mask(src, [transformed_geometry], crop=True)
            out_meta = src.meta.copy()
            
            # Create mask for the geometry
            mask_array = create_mask(transformed_geometry, 
                                   (out_image.shape[1], out_image.shape[2]), 
                                   out_transform)
            
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "nodata": None
            })
            return out_image[0], mask_array, out_meta
    except ValueError as e:
        logger.error(f"Error cropping raster: {str(e)}")
        raise

def calculate_ndvi(nir_array, red_array, mask_array):
    """Calculate NDVI from NIR and RED bands with masking."""
    # Convert to float to avoid integer division
    nir = nir_array.astype(float)
    red = red_array.astype(float)
    
    # Handle potential division by zero
    denominator = (nir + red)
    ndvi = np.where(
        denominator != 0,
        (nir - red) / denominator,
        0
    )
    
    # Clip values to [-1, 1] range and apply mask
    ndvi = np.clip(ndvi, -1, 1)
    # Set areas outside the polygon to nodata (transparent)
    ndvi = np.where(mask_array, ndvi, np.nan)
    return ndvi

def apply_colormap(ndvi_array, mask_array):
    """Apply jet colormap to NDVI values"""
    # Create normalized data in range [0,1] for colormap
    norm = Normalize(vmin=-1, vmax=1)
    normalized_ndvi = norm(ndvi_array)
    
    # Get jet colormap
    jet_colors = cm.jet(normalized_ndvi)
    
    # Convert to uint8 for RGB
    rgb_image = (jet_colors[:, :, :3] * 255).astype(np.uint8)
    
    # Add alpha channel based on mask and valid NDVI values
    alpha = np.where(mask_array & ~np.isnan(ndvi_array), 255, 0).astype(np.uint8)
    
    # Stack RGB and alpha
    rgba_image = np.dstack((rgb_image, alpha))
    
    return rgba_image

def zip_results(files_to_zip, output_zip):
    """Zip specified files."""
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_zip:
            zipf.write(file, os.path.basename(file))

def main():
    # Load JSON configuration
    with open('input.json', 'r') as f:
        config = json.load(f)
    
    try:
        # Download and crop TIFF files
        red_url = config['urls'][0]  # VIS (RED) band
        nir_url = config['urls'][1]  # SWIR band (substitute for NIR)
        
        # Download files
        logger.info("Downloading files...")
        red_file = download_tiff(red_url, "red.tif")
        nir_file = download_tiff(nir_url, "nir.tif")
        
        # Get geometry from config
        geometry = config['polygon']['geometry']
        
        logger.info("Cropping images...")
        # Crop both images
        red_data, mask_array, meta = crop_tiff(red_file, geometry)
        nir_data, _, _ = crop_tiff(nir_file, geometry)
        
        logger.info("Calculating NDVI...")
        # Calculate NDVI with mask
        ndvi = calculate_ndvi(nir_data, red_data, mask_array)
        
        logger.info("Applying colormap...")
        # Apply jet colormap
        colored_ndvi = apply_colormap(ndvi, mask_array)
        
        # Update metadata for RGBA output
        meta.update({
            "dtype": "uint8",
            "count": 4,  # 4 bands for RGBA
            "nodata": None
        })
        
        # Save colored NDVI result
        output_file = "ndvi_colored.tif"
        with rasterio.open(output_file, "w", **meta) as dst:
            for idx in range(4):
                dst.write(colored_ndvi[:, :, idx], idx + 1)
        
        logger.info(f"Colored NDVI saved as {output_file}")
        
        # Cleanup downloaded files
        os.remove(red_file)
        os.remove(nir_file)
        
        # Zip results
        files_to_zip = [output_file]
        zip_filename = "results.zip"
        zip_results(files_to_zip, zip_filename)
        
        logger.info(f"Results zipped in {zip_filename}")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()