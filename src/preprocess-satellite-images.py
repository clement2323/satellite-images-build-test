import os
import sys

import numpy as np
from astrovision.data import SatelliteImage, SegmentationLabeledSatelliteImage

from src.classes.filters.filter import Filter
from src.functions import download_data, labelling

# source = "PLEIADES"
# dep = "MAYOTTE"
# year = "2020"
# n_bands = 3
# type_labeler = "BDTOPO"
# task = "segmentation"
# tiles_size = 250


def main(
    source: str, dep: str, year: str, n_bands: int, type_labeler: str, task: str, tiles_size: int
):
    """
    Main method.
    """
    # 1- download pour (dep/annee)
    download_data.download_data(source, dep, year)

    labeler = labelling.get_labeler(type_labeler, year, dep, task)
    for im in os.listdir(f"data/data-raw/{source}/{dep}/{year}/"):
        # 2- Ouvrir avec SatelliteImage
        si = SatelliteImage.from_raster(
            file_path=os.path.join(f"data/data-raw/{source}/{dep}/{year}/", im),
            n_bands=n_bands,
        )

        # 3- Labeliser avec labeler (labeler/tache)
        label = labeler.create_label(si)
        lsi = SegmentationLabeledSatelliteImage(si, label)

        # 4- Split les tuiles (param tiles_size)
        splitted_lsi = lsi.split(tiles_size)

        # 5- Filtre too black and clouds
        filter_ = Filter()
        is_cloud = filter_.is_cloud(
            lsi.satellite_image,
            tiles_size=tiles_size,
            threshold_center=0.7,
            threshold_full=0.4,
            min_relative_size=0.0125,
        )

        splitted_lsi_filtered = [
            lsi
            for lsi, cloud in zip(splitted_lsi, is_cloud)
            if not (
                filter_.is_too_black(
                    lsi.satellite_image, black_value_threshold=25, black_area_threshold=0.5
                )
                or cloud
            )
        ]

        # 7- save dans data-prepro
        os.makedirs(
            f"data/data-preprocessed/labels/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/",  # noqa
            exist_ok=True,
        )
        for i, lsi in enumerate(splitted_lsi_filtered):
            filename, ext = os.path.splitext(im)
            lsi.satellite_image.to_raster(
                f"data/data-preprocessed/patchs/{task}/{source}/{dep}/{year}/{tiles_size}/{filename}_{i:04d}{ext}"  # noqa
            )
            np.save(
                f"data/data-preprocessed/labels/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/{filename}_{i:04d}.npy",  # noqa
                lsi.label,
            )

        # 8- upload to Minio


if __name__ == "__main__":
    main(
        str(sys.argv[1]),
        str(sys.argv[2]),
        str(sys.argv[3]),
        str(sys.argv[4]),
        str(sys.argv[5]),
        str(sys.argv[6]),
        str(sys.argv[7]),
    )
