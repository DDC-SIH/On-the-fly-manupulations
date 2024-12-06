from osgeo import gdal
import os

def convert_to_cog(input_tif, output_tif):
    """Convert a GeoTIFF to Cloud Optimized GeoTIFF with LZW compression"""
    cog_options = gdal.TranslateOptions(
        format='GTiff',
        creationOptions=[
            'COMPRESS=LZW',
            'TILED=YES',
            'COPY_SRC_OVERVIEWS=YES',
            'BIGTIFF=YES'
        ]
    )
    gdal.Translate(output_tif, input_tif, options=cog_options)

def process_satellite_subdataset(input_h5_file, subdataset, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    vrt_path = os.path.join(output_dir, f'{subdataset}_geos.vrt')
    temp_tif = os.path.join(output_dir, f'{subdataset}_temp.tif')
    final_tif = os.path.join(output_dir, f'{subdataset}_region_cog.tif')
    
    geos_srs = '+proj=geos +h=35782063 +a=6378137.0 +b=6356752.3142 +lon_0=74.16 +no_defs'
    
    translate_options = gdal.TranslateOptions(
        format='VRT',
        outputSRS=geos_srs,
        outputBounds=[-5632000, 5610000, 5632000, -5610000]
    )
    
    input_path = f'HDF5:"{input_h5_file}"://{subdataset}'
    gdal.Translate(vrt_path, input_path, options=translate_options)
    
    # Updated bounding box coordinates
    warp_options = gdal.WarpOptions(
        dstSRS='EPSG:4326',
        outputBounds=[45.5991, -10.1249, 105.8995, 44.4621],  # [West, South, East, North]
        resampleAlg='bilinear',
        srcNodata=0,
        dstNodata=0
    )
    
    gdal.Warp(temp_tif, vrt_path, options=warp_options)
    convert_to_cog(temp_tif, final_tif)
    
    # Cleanup temporary files
    for file in [vrt_path, temp_tif]:
        if os.path.exists(file):
            os.remove(file)

def main():
    input_file = '3RIMG_04SEP2024_1015_L1B_STD_V01R00.h5'
    output_base_dir = 'region_outputs'
    
    image_subdatasets = [
        'IMG_MIR', 'IMG_SWIR', 'IMG_TIR1',
        'IMG_TIR2', 'IMG_VIS', 'IMG_WV'
    ]
    
    try:
        for subdataset in image_subdatasets:
            print(f"Processing {subdataset}...")
            process_satellite_subdataset(input_file, subdataset, output_base_dir)
            print(f"Completed processing {subdataset}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    else:
        print("All processing completed successfully!")

if __name__ == '__main__':
    main()