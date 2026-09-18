from cl_framework.hpo import HyperparameterOptimizer
from cl_framework.train import ContinualLearningTrainer
from cl_framework.config import MODE, STRATEGIES, DATASETS, N_TRIALS
from cl_framework.utils import ParameterLoader
from cl_framework.dataset_loader import DatasetLoader
import argparse
import logging
import time
from typing import List

import os
from itertools import product

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)

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

    def run(self, dataset=None, strategy=None):
        datasets = [dataset] if dataset else self.datasets
        strategies = [strategy] if strategy else self.strategies

        for dataset in datasets:
            try:
                loader = DatasetLoader(dataset).load()  # For local pc
                # loader = DatasetLoader(dataset).load_from_tensor()  # For the server
                benchmark, hpo_benchmark, input_size, n_labels, n_experiences = loader.to_tuple()

            except Exception as e:
                logger.error(f"Failed to load dataset '{dataset}': {e}")
                continue

            for strategy in strategies:
                start_time = time.time()
                logger.info("=" * 60)
                logger.info("EXPERIMENT START")
                logger.info(f"Dataset  : {dataset.upper()}")
                logger.info(f"Strategy : {strategy.upper()}")
                logger.info(f"Mode     : {self.mode.upper()}")
                logger.info("=" * 60)

                try:
                    self._run_strategy(
                        strategy,
                        dataset,
                        benchmark,
                        hpo_benchmark,
                        input_size,
                        n_labels,
                        n_experiences
                    )
                    elapsed = time.time() - start_time

                    logger.info("=" * 60)
                    logger.info("EXPERIMENT COMPLETED")
                    logger.info(f"Dataset  : {dataset.upper()}")
                    logger.info(f"Strategy : {strategy.upper()}")
                    logger.info(f"Duration : {elapsed / 60:.2f} minutes")
                    logger.info("=" * 60)
                    
                except Exception as e:
                    logger.error(f"Error during strategy '{strategy}': {e}")
                
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



#if __name__ == "__main__":
#    experiments = CLExperiments()
#    experiments.run()

# to get the input arguments from the command line, we can use argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--strategy", type=str)
    args = parser.parse_args()

    experiments = CLExperiments()

    if args.dataset and args.strategy:
        experiments.run(
            dataset=args.dataset,
            strategy=args.strategy
        )

    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

        combinations = list(
            product(experiments.datasets, experiments.strategies)
        )

        dataset, strategy = combinations[task_id]

        print(f"SLURM task {task_id}")
        print(f"Dataset: {dataset}")
        print(f"Strategy: {strategy}")

        experiments.run(
            dataset=dataset,
            strategy=strategy
        )

    else:
        experiments.run()
    
