import h5py
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
    Extract and project L2C subdatasets from HDF5 file using Mercator projection
    """
    # L2C specific subdatasets
    L2C_DATASETS = ['DHI', 'DNI', 'GHI', 'INS']
    
    # Projection parameters from metadata
    proj_params = {
        'proj': 'merc',
        'lon_0': 77.25,  # Projection_Information_longitude_of_projection_origin
        'lat_ts': 17.75, # Projection_Information_standard_parallel
        'x_0': 0,        # Projection_Information_false_easting
        'y_0': 0,        # Projection_Information_false_northing
        'a': 6378137,    # Projection_Information_semi_major_axis
        'b': 6356752.3142, # Projection_Information_semi_minor_axis
        'units': 'm'
    }

    crs = CRS.from_dict(proj_params)
    
    # Bounds from metadata
    bounds = {
        'left': 44.5,    # left_longitude
        'right': 110.0,  # right_longitude
        'bottom': -10.0, # lower_latitude
        'top': 45.5      # upper_latitude
    }

    transformer = Transformer.from_crs(
        CRS.from_epsg(4326),
        crs,
        always_xy=True
    )

    left, bottom = transformer.transform(bounds['left'], bounds['bottom'])
    right, top = transformer.transform(bounds['right'], bounds['top'])

    with h5py.File(h5_file_path, 'r') as h5f:
        for dataset in L2C_DATASETS:
            try:
                if dataset not in h5f:
                    logger.warning(f"Skipping {dataset} - not found in file")
                    continue
                    
                logger.info(f"Processing {dataset}")
                
                data = h5f[dataset][:]
                data = np.squeeze(data)
                
                if len(data.shape) != 2:
                    logger.warning(f"Skipping {dataset} - unexpected shape {data.shape}")
                    continue
                
                logger.info(f"Data shape: {data.shape}")
                
                # Handle fill values
                fill_value = h5f[dataset].attrs.get(f'{dataset}__FillValue', -999)
                data = np.ma.masked_equal(data, fill_value)
                
                output_path = f"{output_dir}/{dataset}.tif"
                
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
                    nodata=fill_value
                ) as dst:
                    dst.write(data.filled(fill_value), 1)
                    dst.update_tags(**{
                        'LONG_NAME': h5f[dataset].attrs.get(f'{dataset}_long_name', ''),
                        'STANDARD_NAME': h5f[dataset].attrs.get(f'{dataset}_standard_name', ''),
                        'UNITS': h5f[dataset].attrs.get(f'{dataset}_units', ''),
                        'GRID_MAPPING': h5f[dataset].attrs.get(f'{dataset}_grid_mapping', '')
                    })
                logger.info(f"Successfully written {output_path}")
                
            except Exception as e:
                logger.error(f"Error processing {dataset}: {str(e)}")
                continue

def main():
    h5_file = "3RIMG_04SEP2024_1015_L2C_INS_V01R00.h5"
    output_dir = "l2c_projected_data"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    extract_and_project_subdatasets(h5_file, output_dir)

if __name__ == "__main__":
    main()