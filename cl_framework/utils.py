import json
from pathlib import Path


class ParameterLoader:
    def __init__(self, base_dir="results"):
        self.base_dir = Path(base_dir)

    def get_best_params_path(self, dataset_name: str, strategy_name: str) -> Path:
        return self.base_dir / dataset_name / strategy_name / "best_params.json"

    def load_best_params(self, dataset_name: str, strategy_name: str):
        path = self.get_best_params_path(dataset_name, strategy_name)
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return None

    def has_best_params(self, dataset_name: str, strategy_name: str) -> bool:
        return self.get_best_params_path(dataset_name, strategy_name).exists()