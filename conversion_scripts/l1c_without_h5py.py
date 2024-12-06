import netCDF4
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pyproj import CRS, Transformer
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_and_project_subdatasets(h5_file_path, output_dir):
    """
    Extract and project base image subdatasets from HDF5 file using Mercator projection
    """
    # Base image keys to process
    BASE_IMAGES = ['IMG_MIR', 'IMG_SWIR', 'IMG_TIR1', 'IMG_TIR2', 'IMG_VIS', 'IMG_WV']
    
    proj_params = {
        'proj': 'merc',
        'lon_0': 77.25,
        'lat_ts': 17.75,
        'x_0': 0,
        'y_0': 0,
        'a': 6378137,
        'b': 6356752.3142,
        'units': 'm'
    }

    crs = CRS.from_dict(proj_params)
    
    bounds = {
        'left': 44.5,
        'right': 110.0,
        'bottom': -10.0,
        'top': 45.5
    }

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        crs,
        always_xy=True
    )

    left, bottom = transformer.transform(bounds['left'], bounds['bottom'])
    right, top = transformer.transform(bounds['right'], bounds['top'])

    with netCDF4.Dataset(h5_file_path, 'r') as h5f:
        for key in BASE_IMAGES:
            try:
                if key not in h5f.variables:
                    logger.warning(f"Skipping {key} - not found in file")
                    continue
                    
                logger.info(f"Processing {key}")
                
                data = h5f.variables[key][:]
                data = np.squeeze(data)
                
                if len(data.shape) != 2:
                    logger.warning(f"Skipping {key} - unexpected shape {data.shape}")
                    continue
                
                logger.info(f"Data shape: {data.shape}")
                
                # Read scale factor and offset
                scale_factor = h5f.variables[key].getncattr(f'{key}_lab_radiance_scale_factor') if f'{key}_lab_radiance_scale_factor' in h5f.variables[key].ncattrs() else 1.0
                add_offset = h5f.variables[key].getncattr(f'{key}_lab_radiance_add_offset') if f'{key}_lab_radiance_add_offset' in h5f.variables[key].ncattrs() else 0.0
                
                data = data * scale_factor + add_offset
                
                output_path = os.path.join(output_dir, f"{key}.tif")
                
                transform = from_bounds(
                    left, bottom, right, top,
                    data.shape[1], data.shape[0]
                )
                
                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    crs=crs.to_wkt(),
                    transform=transform,
                ) as dst:
                    dst.write(data, 1)
                    dst.update_tags(**{
                        'WAVELENGTH': h5f.variables[key].getncattr(f'{key}_central_wavelength') if f'{key}_central_wavelength' in h5f.variables[key].ncattrs() else '',
                        'UNITS': h5f.variables[key].getncattr(f'{key}_RADIANCE_units') if f'{key}_RADIANCE_units' in h5f.variables[key].ncattrs() else ''
                    })
                logger.info(f"Successfully written {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {key}: {str(e)}")
                continue

if __name__ == "__main__":
    h5_file = "3RIMG_04SEP2024_1015_L1C_ASIA_MER_V01R00.h5"
    output_dir = "projected_data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    extract_and_project_subdatasets(h5_file, output_dir)
