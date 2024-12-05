from osgeo import gdal
import h5py
import json
import numpy as np
from datetime import datetime
import os


def convert_attribute_value(value):
    """Convert HDF5 attribute value to JSON serializable format."""
    if isinstance(value, (np.ndarray, list)):
        if len(value) == 1:
            return value[0].item() if isinstance(value[0], np.generic) else value[0]
        return [item.item() if isinstance(item, np.generic) else item for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def extract_h5_metadata(h5_file_path):
    """Extract metadata from HDF5 file."""
    metadata = {}

    with h5py.File(h5_file_path, "r") as f:
        metadata["root_attributes"] = {
            attr_name: convert_attribute_value(f.attrs[attr_name])
            for attr_name in f.attrs.keys()
        }

        metadata["datasets"] = {}

        def extract_dataset_info(name, obj):
            if isinstance(obj, h5py.Dataset):
                metadata["datasets"][name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "attributes": {
                        key: convert_attribute_value(obj.attrs[key])
                        for key in obj.attrs.keys()
                    },
                }

        f.visititems(extract_dataset_info)
        metadata["file_info"] = {
            "filename": h5_file_path,
            "extracted_date": datetime.now().isoformat(),
        }

    return metadata


def extract_tiff_metadata(tiff_file_path):
    """Extract metadata from TIFF file using GDAL."""
    return gdal.Info(tiff_file_path, format="json", stats=True)


def save_json_metadata(metadata, output_file):
    """Save metadata dictionary to JSON file."""
    with open(output_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def cleanup_tiff_files(files):
    """Clean up temporary TIFF files."""
    for file in files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"Removed: {file}")
        except OSError as e:
            print(f"Error removing {file}: {e}")


def cleanup_aux_files(files):
    """Clean up auxiliary XML files."""
    for file in files:
        aux_file = f"{file}.aux.xml"
        try:
            if os.path.exists(aux_file):
                os.remove(aux_file)
                print(f"Removed: {aux_file}")
        except OSError as e:
            print(f"Error removing {aux_file}: {e}")


def process_files(input_file, bands, georef_params, webmercator_srs, optimized_params):
    """Process HDF5 and generate TIFF files with metadata, then clean up."""
    # Extract and save HDF5 metadata
    h5_metadata = extract_h5_metadata(input_file)
    save_json_metadata(h5_metadata, "h5_metadata.json")
    print("Saved HDF5 metadata")

    # Process each band
    for band_name, dataset_path in bands.items():
        temp_files = []  # Track files for cleanup

        try:
            # Extract band
            output_tif = f"IMG_{band_name}.tif"
            band_ds = gdal.Open(f'HDF5:"{input_file}":{dataset_path}')
            gdal.Translate(output_tif, band_ds, format="GTiff")
            band_ds = None
            temp_files.append(output_tif)

            # Georeference
            georef_tif = f"IMG_{band_name}_georef.tif"
            gdal.Translate(
                georef_tif,
                output_tif,
                outputBounds=[
                    georef_params["ulx"],
                    georef_params["uly"],
                    georef_params["lrx"],
                    georef_params["lry"],
                ],
                outputSRS=georef_params["srs"],
            )
            temp_files.append(georef_tif)

            # Reproject to Web Mercator
            webmercator_tif = f"IMG_{band_name}_webmercator.tif"
            gdal.Warp(webmercator_tif, georef_tif, dstSRS=webmercator_srs)
            temp_files.append(webmercator_tif)

            # Optimize for Google Maps
            optimized_tif = f"IMG_{band_name}_optimized.tif"
            gdal.Translate(
                optimized_tif, webmercator_tif, creationOptions=optimized_params
            )
            temp_files.append(optimized_tif)

            # Extract and save metadata for each processing stage
            tiff_metadata = {
                "original": extract_tiff_metadata(output_tif),
                "georeferenced": extract_tiff_metadata(georef_tif),
                "webmercator": extract_tiff_metadata(webmercator_tif),
                "optimized": extract_tiff_metadata(optimized_tif),
            }

            save_json_metadata(tiff_metadata, f"metadata_{band_name}.json")
            print(f"Saved metadata for band {band_name}")

        finally:
            # Cleanup TIFF files regardless of success or failure
            cleanup_tiff_files(temp_files)
            cleanup_aux_files(temp_files)


def main():
    input_file = "3RIMG_04SEP2024_1015_L1C_ASIA_MER_V01R00.h5"
    bands = {
        "VIS": "//IMG_VIS",
        "MIR": "//IMG_MIR",
        "SWIR": "//IMG_SWIR",
        "TIR1": "//IMG_TIR1",
        "TIR2": "//IMG_TIR2",
        "WV": "//IMG_WV",
    }
    georef_params = {
        "ulx": 44.5,
        "uly": 48.1,
        "lrx": 110,
        "lry": -7.4,
        "srs": "EPSG:4326",
    }
    webmercator_srs = "EPSG:3857"
    optimized_params = ["TILED=YES", "COMPRESS=DEFLATE"]

    try:
        process_files(
            input_file, bands, georef_params, webmercator_srs, optimized_params
        )
        print("Processing and cleanup completed successfully")
    except Exception as e:
        print(f"Error during processing: {str(e)}")


if __name__ == "__main__":
    main()
