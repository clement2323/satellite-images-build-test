import os
import sys

from astrovision.data import SatelliteImage

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
    for im in os.listdir(f"data/data-raw/{source}/{dep}/{year}/")[0]:
        # 2- Ouvrir avec SatelliteImage
        si = SatelliteImage.from_raster(
            file_path=os.path.join(f"data/data-raw/{source}/{dep}/{year}/", im),
            n_bands=n_bands,
        )

        # 3- Labeliser avec labeler (labeler/tache)
        label = labeler.create_label(si)

        # 4- Split les tuiles (param tiles_size)
        splitted_si = si.split(tiles_size)

        # NOOOON CA VA PAS IL FAUT FAIRE LE FILTER CLOUD SUR LA GROSSE IMAGE
        # 5- Filtre too black and clouds
        filter_ = Filter()
        splitted_si_filtered = [
            si
            for si in splitted_si
            if not (
                filter_.is_too_black(si, black_value_threshold=25, black_area_threshold=0.5)
                or filter_.is_cloud(
                    si,
                    tiles_size=tiles_size,
                    threshold_center=0.7,
                    threshold_full=0.4,
                    min_relative_size=0.0125,
                )
            )
        ]

        # 7- save dans data-prepro

        # temp
        return label, splitted_si_filtered


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
