
from pathlib import Path
import yaml
DATASETS_CONFIG = "datasets.yaml"

def dataset_paths(config_path:str=DATASETS_CONFIG):
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / DATASETS_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    root = Path(config.get("data_root", "")).expanduser()
    datasets = config.get("datasets", {})

    return {
        k: (root / v).resolve() if not Path(v).is_absolute() else Path(v).resolve()
        for k, v in datasets.items()
    }