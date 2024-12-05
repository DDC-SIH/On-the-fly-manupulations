from osgeo import gdal

# File paths
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

# Georeference bands and reproject to Web Mercator
for band_name, dataset_path in bands.items():
    # Extract band
    output_tif = f"IMG_{band_name}.tif"
    band_ds = gdal.Open(f'HDF5:"{input_file}":{dataset_path}')
    gdal.Translate(output_tif, band_ds, format="GTiff")
    band_ds = None  # Close dataset

    # Georeference
    georef_tif = f"IMG_{band_name}_georef.tif"
    gdal.Translate(
        georef_tif,
        output_tif,
        outputBounds=[georef_params["ulx"], georef_params["uly"], georef_params["lrx"], georef_params["lry"]],
        outputSRS=georef_params["srs"]
    )

    # Reproject to Web Mercator
    webmercator_tif = f"IMG_{band_name}_webmercator.tif"
    gdal.Warp(webmercator_tif, georef_tif, dstSRS=webmercator_srs)

    # Optimize for Google Maps
    optimized_tif = f"IMG_{band_name}_optimized.tif"
    gdal.Translate(optimized_tif, webmercator_tif, creationOptions=optimized_params)

# Verify results
for band_name in bands.keys():
    optimized_tif = f"IMG_{band_name}_optimized.tif"
    info = gdal.Info(optimized_tif, format="json")
    print(f"Info for {optimized_tif}:")
    print(info)
