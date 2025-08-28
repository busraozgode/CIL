from cl_framework.hpo import HyperparameterOptimizer
from cl_framework.train import ContinualLearningTrainer
from cl_framework.config import MODE, STRATEGIES, DATASETS, N_TRIALS
from cl_framework.utils import ParameterLoader
from cl_framework.dataset_loader import DatasetLoader
import logging
from typing import List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CLExperiments:
    def __init__(
        self,
        mode: str = MODE,
        strategies: List[str] = STRATEGIES,
        datasets: List[str] = DATASETS,
        n_trials: int = N_TRIALS
    ):
        self.mode = mode
        self.strategies = strategies
        self.datasets = datasets
        self.n_trials = n_trials
        self.param_loader = ParameterLoader()

    def run(self):
        for dataset in self.datasets:
            try:
                loader = DatasetLoader(dataset).load() #For local pc
                #loader = DatasetLoader(dataset).load_from_tensor() #For the server
                benchmark, hpo_benchmark, input_size, n_labels, n_experiences = loader.to_tuple()
            except Exception as e:
                logger.error(f" Failed to load dataset '{dataset}': {e}")
                continue
            
            for strategy in self.strategies:
                logger.info(f" Running {strategy.upper()} on {dataset.upper()} in {self.mode.upper()} mode")
                try:
                    self._run_strategy(
                        strategy, dataset, benchmark, hpo_benchmark, input_size, n_labels, n_experiences
                    )
                except Exception as e:
                    logger.error(f" Error during strategy '{strategy}': {e}")

    def _run_strategy(self, strategy: str, dataset: str, benchmark, hpo_benchmark, input_size: int, n_labels: int, n_experiences: int):
        if self.mode == "hpo":
            hpo_optimizer = HyperparameterOptimizer(hpo_benchmark, input_size, n_labels, strategy, dataset, n_trials=self.n_trials, optimize_metric="Acc") #Loss,MacroF1,Acc.
            best_params = hpo_optimizer.run()
            if best_params is None:
                logger.error(" HPO failed to find best parameters.")
                return
        elif self.mode in ["train", "auto"]:
            best_params = self.param_loader.load_best_params(dataset, strategy)

            if best_params is None:
                if self.mode == "train":
                    logger.warning(" No best parameters found. Please run HPO first.")
                    return
                logger.info(" No best parameters found. Running HPO first...")
                hpo_optimizer = HyperparameterOptimizer(hpo_benchmark, input_size, n_labels, strategy, dataset, n_trials=self.n_trials, optimize_metric="Acc")
                best_params = hpo_optimizer.run()

            trainer = ContinualLearningTrainer(benchmark, input_size, n_labels, n_experiences, strategy, dataset, best_params)
            trainer.train()
        else:
            raise ValueError(f" Invalid MODE: {self.mode}")



if __name__ == "__main__":
    experiments = CLExperiments()
    experiments.run()