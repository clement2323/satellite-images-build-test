import os

from astrovision.data import SatelliteImage

from src.classes.filters.filter import Filter
from src.functions import download_data, labelling

source = "PLEIADES"
dep = "MAYOTTE"
year = "2020"
n_bands = 3
type_labeler = "BDTOPO"
task = "segmentation"
tiles_size = 250
# 1- download pour (dep/annee)
download_data.download_data(source, dep, year)

# 2- Ouvrir avec SatelliteImage
x = os.listdir(f"data/data-raw/{source}/{dep}/{year}/")

si = SatelliteImage.from_raster(
    file_path=os.path.join(f"data/data-raw/{source}/{dep}/{year}/", x[0]),
    n_bands=n_bands,
)

# 3- Labeliser avec labeler (labeler/tache)
labeler = labelling.get_labeler(type_labeler, year, dep, task)
label = labeler.create_label(si)


# 3.5- Creation masque nuages pour Pleiades et split masque
# TODO
filter_ = Filter()

# 4- Split les tuiles (param tiles_size)
splitted_si = si.split(250)

# 5- Filtre too black

splitted_si_filtered = [
    si
    for si in splitted_si
    if not filter_.is_too_black(si, black_value_threshold=25, black_area_threshold=0.5)
]

# 6- filtre nuages pour pleiades

# 7- save dans data-prepro
