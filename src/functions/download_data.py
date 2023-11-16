import os

import s3fs


def download_data(source: str, dep: str, year: str):
    """
    Download data from a specified source, department, and year.

    Parameters:
    - source (str): The data source identifier.
    - dep (str): The department identifier.
    - year (str): The year for which data should be downloaded.

    """

    print("\n*** 1- Téléchargement des données...\n")
    s3_path = f"projet-slums-detection/data-raw/{source}/{dep}/{year}/"
    local_path = f"../data-raw/{source}/{dep}/{year}/"

    try:
        # Initialize S3 file system
        fs = s3fs.S3FileSystem(
            client_kwargs={"endpoint_url": "https://" + "minio.lab.sspcloud.fr"},
            key=os.getenv("AWS_ACCESS_KEY_ID"),
            secret=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        # Download data from S3 to local path
        fs.download(rpath=s3_path, lpath=local_path, recursive=True)

        print("\n*** Téléchargement terminé !\n")

    except FileNotFoundError:
        print(f"Error: The specified data path '{s3_path}' does not exist on the S3 bucket.")
        raise

    except Exception as e:
        print(f"Error: An unexpected error occurred during the download process. {e}")
        raise
