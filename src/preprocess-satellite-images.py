import os
import sys

import numpy as np
from astrovision.data import SatelliteImage, SegmentationLabeledSatelliteImage
from tqdm import tqdm

from classes.filters.filter import Filter
from functions import download_data, labelling


def main(
    source: str, dep: str, year: str, n_bands: int, type_labeler: str, task: str, tiles_size: int
):
    """
    Main method.
    """
    # Initialize S3 file system
    fs = download_data.get_file_system()

    print("\n*** 1- Téléchargement de la base d'annotation...\n")
    labeler = labelling.get_labeler(type_labeler, year, dep, task)

    os.makedirs(
        f"data/data-preprocessed/labels/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/",  # noqa
        exist_ok=True,
    )

    print("\n*** 2- Annotation, découpage et filtrage des images...\n")
    for im in tqdm(fs.ls(f"projet-slums-detection/data-raw/{source}/{dep}/{year}/")):
        # 1- Ouvrir avec SatelliteImage
        si = SatelliteImage.from_raster(
            file_path=f"/vsis3/{im}",
            n_bands=int(n_bands),
        )

        # 2- Labeliser avec labeler (labeler/tache)
        label = labeler.create_label(si)
        lsi = SegmentationLabeledSatelliteImage(si, label)

        # 3- Split les tuiles (param tiles_size)
        splitted_lsi = lsi.split(int(tiles_size))

        # 4- Filtre too black and clouds
        filter_ = Filter()

        if source == "PLEIADES":
            is_cloud = filter_.is_cloud(
                lsi.satellite_image,
                tiles_size=int(tiles_size),
                threshold_center=0.7,
                threshold_full=0.4,
                min_relative_size=0.0125,
            )
        else:
            is_cloud = [0] * len(splitted_lsi.satellite_image)

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

        # 5- save dans data-prepro
        for i, lsi in enumerate(splitted_lsi_filtered):
            filename, ext = os.path.splitext(os.path.basename(im))
            lsi.satellite_image.to_raster(
                f"data/data-preprocessed/patchs/{task}/{source}/{dep}/{year}/{tiles_size}/{filename}_{i:04d}{ext}"  # noqa
            )
            np.save(
                f"data/data-preprocessed/labels/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/{filename}_{i:04d}.npy",  # noqa
                lsi.label,
            )

    print("\n*** 3- Preprocessing terminé !\n")


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
