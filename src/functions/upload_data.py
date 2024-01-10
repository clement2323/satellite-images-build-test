import yaml
from functions.download_data import get_file_system


def upload_normalization_metrics(
    metrics: dict,
    source: str,
    dep: str,
    year: str,
    task: str,
    tiles_size: int,
) -> None:
    """
    Upload a Yaml file to s3 containing mean and standard deviation per channel
    """
    # Initialize S3 file system
    fs = get_file_system()

    yaml_data = yaml.dump(metrics, default_flow_style=False)

    # Save metrics file in s3
    with fs.open(
        f"projet-slums-detection/data-preprocessed/patchs/{task}/{source}/{dep}/{year}/{tiles_size}/metrics-normalization.yaml",
        "wb",
    ) as f:
        f.write(yaml_data.encode("utf-8"))
