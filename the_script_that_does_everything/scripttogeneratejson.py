from osgeo import gdal
import h5py
import json
import numpy as np
from datetime import datetime
import os
import glob

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

def process_tiff_metadata(tiff_file):
    """Extract and save metadata for a single TIFF file."""
    try:
        metadata = extract_tiff_metadata(tiff_file)
        json_file = f"{os.path.splitext(tiff_file)[0]}_metadata.json"
        save_json_metadata(metadata, json_file)
        print(f"Generated metadata for: {tiff_file}")
        return True
    except Exception as e:
        print(f"Error processing metadata for {tiff_file}: {e}")
        return False

def process_metadata(input_h5_file, bands):
    """Process metadata for H5 file and generated TIFFs."""
    # Generate H5 metadata
    try:
        h5_metadata = extract_h5_metadata(input_h5_file)
        save_json_metadata(h5_metadata, "h5_metadata.json")
        print("Saved H5 metadata")
    except Exception as e:
        print(f"Error extracting H5 metadata: {e}")

    # Process metadata for existing TIFF files
    for band_name in bands.keys():
        tiff_pattern = f"IMG_{band_name}*.tif"
        tiff_files = glob.glob(tiff_pattern)
        
        for tiff_file in tiff_files:
            process_tiff_metadata(tiff_file)

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

    try:
        process_metadata(input_file, bands)
        print("Metadata extraction completed successfully")
    except Exception as e:
        print(f"Error during metadata extraction: {str(e)}")

if __name__ == "__main__":
    main()