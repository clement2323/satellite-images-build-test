import os

import geopandas as gpd
from s3fs import S3FileSystem


def get_file_system() -> S3FileSystem:
    """
    Return the s3 file system.
    """
    return S3FileSystem(
        client_kwargs={"endpoint_url": f"https://{os.environ['AWS_S3_ENDPOINT']}"},
        key=os.environ["AWS_ACCESS_KEY_ID"],
        secret=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def load_bdtopo(
    year: str,
    dep: str,
) -> gpd.GeoDataFrame:
    """
    Load BDTOPO for a given datetime.

    Args:
        year (Literal): Year.
        dep (Literal): Departement.

    Returns:
        gpd.GeoDataFrame: BDTOPO GeoDataFrame.
    """

    if int(year) >= 2019:
        couche, ext = ("BATIMENT", "shp")
    elif int(year) < 2019:
        couche, ext = ("BATI_INDIFFERENCIE", "SHP")

    fs = get_file_system()

    s3_path = f"projet-slums-detection/data-label/BDTOPO/{dep}/{year}/{couche}.*"
    local_path = f"data/data-label/BDTOPO/{dep}/{year}/"

    fs.download(
        rpath=s3_path,
        lpath=local_path,
        recursive=True,
    )

    df = gpd.read_file(f"{local_path}{couche}.{ext}")

    return df
