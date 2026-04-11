from pathlib import Path
import yaml


# -----------------------------
# Load config
# -----------------------------
def load_config(config_path: str) -> dict:
    path = Path(config_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------
# Extract paths (RETURN Path objects)
# -----------------------------
def extract_paths(obj, root: Path):
    paths = []

    if isinstance(obj, dict):
        for v in obj.values():
            paths.extend(extract_paths(v, root))

    elif isinstance(obj, list):
        for v in obj:
            paths.extend(extract_paths(v, root))

    elif isinstance(obj, str):
        paths.append((root / obj).resolve()) 

    return paths


# -----------------------------
# Dataset loader
# -----------------------------
def get_dataset_paths(config: dict, dataset_names: list[str]) -> dict:
    root = Path(config.get("data_root", "")).expanduser().resolve()

    all_paths = {}

    for name in dataset_names:
        dataset_cfg = config.get("datasets", {}).get(name)

        if dataset_cfg is None:
            raise ValueError(f"Unknown dataset: {name}")

        paths = extract_paths(dataset_cfg, root)

        all_paths[name] = paths

    return all_paths


# -----------------------------
# Path sheets
# -----------------------------
def get_path_sheets(config: dict):
    root = Path(config.get("data_root", "")).expanduser().resolve()

    ps_cfg = config.get("path_sheets", [])

    paths = extract_paths(ps_cfg, root)

    return paths


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    config = load_config("~/Documents/video_base_system/config/datasets.yaml")

    dataset_paths = get_dataset_paths(config, ["thermal"])
    path_sheets = get_path_sheets(config)

    print("\nFinal dataset paths:")
    print({k: [str(p) for p in v] for k, v in dataset_paths.items()})

    print("\nFinal path sheets:")
    print([str(p) for p in path_sheets])