from pathlib import Path
import yaml

from typing import List

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
def get_excel_files(folder: Path) -> List[Path]:
    folder = Path(folder)
    print(f"Looking for Excel files in: {folder}")
    if not folder.exists():
        return []

    return list(folder.glob("*.xlsx")) + list(folder.glob("*.csv"))  

def get_path_sheets(config: dict) -> List[Path]:
    root = Path(config.get("data_root", "")).expanduser().resolve()
    paths = extract_paths(config.get("path_sheets", []), root)

    return [
        f
        for folder in paths
        for f in get_excel_files(folder)
    ]
    
def get_all_sheets(config):
    paths = get_path_sheets(config)
    return [str(p) for p in paths]

def get_base_path(config) -> Path:
    return Path(config.get("data_root", "")).expanduser().resolve().parent
import re

def expand(pattern:str ="Image[1-3]") -> List[str]:
    match = re.match(r"(.*)\[(\d+)-(\d+)\]", pattern)
    if match:
        base, start, end = match.groups()
        return [f"{base}{i}" for i in range(int(start), int(end)+1)]
    return [pattern]



if __name__ == "__main__":
    sheets = get_all_sheets()
    print("Path sheets found:")
    for sheet in sheets:
        print(f" - {sheet}")
    root = get_base_path()
    print(f"Base path: {root}")
    img_dirs = expand("folderImage")


