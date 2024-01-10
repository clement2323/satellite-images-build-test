import os

import subprocess

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


def get_raw_images(
    from_s3: bool,
    source: str,
    dep: str,
    year: str,
):
    if int(from_s3):
        fs = get_file_system()

        images = fs.ls((f"projet-slums-detection/data-raw/" f"{source}/{dep}/{year}"))
    else:
        images_path = f"data/data-raw/{source}/{dep}/{year}"
        download_data(images_path, source, dep, year)
        images = [f"{images_path}/{filename}" for filename in os.listdir(images_path)]

    return images


def get_roi(
    dep: str,
):
    fs = get_file_system()
    roi = gpd.read_file(fs.open(f"projet-slums-detection/data-roi/{dep}.geojson", "rb"))

    return roi


def download_data(
    images_path: str,
    source: str,
    dep: str,
    year: str,
):
    """
    Download data from a specified source, department, and year.
    Parameters:
        - source (str): The data source identifier.
        - dep (str): The department identifier.
        - year (str): The year for which data should be downloaded.
    """
    all_exist = os.path.exists(f"{images_path}")

    if all_exist:
        return None

    image_cmd = [
        "mc",
        "cp",
        "-r",
        f"s3/projet-slums-detection/data-raw/{source}/{dep}/{year}/",  # noqa
        f"{images_path}",
    ]

    # download raw images
    subprocess.run(image_cmd, check=True)
