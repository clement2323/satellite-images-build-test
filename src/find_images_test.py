import os
import matplotlib.pyplot as plt
import yaml

from functions import download_data, labelling
from functions.plot_utils import plot_images_mask_around_point

# Mayotte
# dep = "MAYOTTE"
# point = [-12.774895, 45.218719]
# point = [-12.719838, 45.117112]
# point = [-12.961326, 45.126497]

# # Martinique
# dep = "MARTINIQUE"
# point = [14.650193, -61.055606]
# point = [14.574659, -60.975153]

# Guadeloupe
dep = "GUADELOUPE"
# point = [16.205556577017383, -61.482986085610385]
# point = [16.329284496925418, -61.46031328436647]
# point = [16.009606762305808, -61.680187477540215]

# Guyane
dep = "GUYANE"
point = [4.858333, -52.279172]

# Reunion
dep = "REUNION"
# point = [-20.902806594883593, 55.50138840828117]
# point = [-21.05190440896064, 55.223939457016336]
# point = [-21.106326150694997, 55.294520868581344]


source = "PLEIADES"
year = "2022"
task = "segmentation"
type_labeler = "BDTOPO"
tiles_size = "250"
n_bands = "3"
# Initialize S3 file system
fs = download_data.get_file_system()

print("\n*** 1- Téléchargement de la base d'annotation...\n")
labeler = labelling.get_labeler(type_labeler, year, dep, task)

os.makedirs(
    f"data/data-preprocessed/labels/{type_labeler}/{task}/{source}/{dep}/{year}/{tiles_size}/",  # noqa
    exist_ok=True,
)

os.makedirs(
    f"data/images-test-{dep.lower()}/",  # noqa
    exist_ok=True,
)

plt.close()
images, bb_images = plot_images_mask_around_point(
    point, source, dep, year, labeler, n_bands, fs, nb_dist=1
)
images.savefig(f"data/images-test-{dep.lower()}/images_{point}_{dep}_{year}.png")


def bounding_boxes_to_yaml(bounding_boxes, dep):
    # Create a dictionary with the bounding boxes
    data = {dep: bounding_boxes}

    # Convert the dictionary to a YAML formatted string
    yaml_data = yaml.dump(data, default_flow_style=False)

    return yaml_data


# Example usage
bounding_boxes = [[10, 20, 30, 40], [50, 60, 70, 80]]
yaml_output = bounding_boxes_to_yaml(bounding_boxes)
print(yaml_output)

with open("bb_test.yaml", "w") as file:
    file.write(yaml_output)

# ecrire à la suite du fichier et ne pas en créer un nouveau à chaque fois


def load_bounding_boxes_from_yaml(file_path, dep):
    # Open the YAML file and read the data
    with open(file_path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    # Extract the bounding boxes list
    bounding_boxes = data.get(dep, [])

    return bounding_boxes
