import torch
import json
import os
from pathlib import Path
from optuna.importance import get_param_importances
import optuna
from torch.optim import Adam

from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics, forgetting_metrics,
    timing_metrics, confusion_matrix_metrics
)

from cl_framework.strategies import StrategyFactory
from cl_framework.config import get_hpo_param_defs
from cl_framework.simple_mlp import MLPFactory
from cl_framework.StandaloneMetrics import MacroF1PluginMetric


class HyperparameterOptimizer:
    def __init__(self, benchmark, input_size, n_labels, strategy_name, dataset_name, n_trials=10, optimize_metric="Acc"):
        self.benchmark = benchmark
        self.input_size = input_size
        self.n_labels = n_labels
        self.strategy_name = strategy_name
        self.dataset_name = dataset_name
        self.n_trials = n_trials

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimize_metric = optimize_metric
        project_root = Path(__file__).parent.parent
        self.results_dir = project_root / 'Results' / dataset_name.lower() / strategy_name
        os.makedirs(self.results_dir, exist_ok=True)

    def _suggest_params(self, trial):
        param_defs = get_hpo_param_defs(self.strategy_name)
        params = {}

        for name, settings in param_defs.items():
            ptype = settings["type"]
            if ptype == "int":
                params[name] = trial.suggest_int(name, settings["low"], settings["high"])
            elif ptype == "float":
                params[name] = trial.suggest_float(name, settings["low"], settings["high"], log=settings.get("log", False))
            elif ptype == "categorical":
                params[name] = trial.suggest_categorical(name, settings["choices"])
            else:
                raise ValueError(f"Unsupported HPO param type: {ptype}")
        return params

    def _objective(self, trial):
        hpo_params = self._suggest_params(trial)
        print(f"Trial {trial.number} optimized hyperparameters: {hpo_params}")
        
        model = MLPFactory().create(input_size=self.input_size, num_classes=self.n_labels)
        
        optimizer = Adam(model.parameters(), lr=0.00033)  # Learning rate could be part of hpo_params

        eval_plugin = EvaluationPlugin(
            accuracy_metrics(epoch=True, experience=True, stream=True),
            loss_metrics(epoch=True, experience=True, stream=True),
            timing_metrics(epoch=True),
            forgetting_metrics(experience=True, stream=True),
            MacroF1PluginMetric(eval_phase=True, epoch=False, experience=True, stream=True),
            confusion_matrix_metrics(
                num_classes=self.benchmark.n_classes, save_image=False, stream=True
            ),
            loggers=[InteractiveLogger(), TensorboardLogger()]
        )

        strategy = StrategyFactory.get_strategy(
            self.strategy_name, model, optimizer, eval_plugin, self.device, **hpo_params
        )

        results = []
        for exp in self.benchmark.train_stream[:3]:  # limit to 3 for speed
            labels_in_exp = set([y for _, y, *_ in exp.dataset])
            print(f"\nTraining on experience {exp.current_experience} with labels: {sorted(labels_in_exp)}")
            strategy.train(exp)
            print("Training completed. Evaluating...")
            results.append(strategy.eval(self.benchmark.test_stream))

        last_result = results[-1]
        metric_key_map = {
        "Acc": "Top1_Acc_Stream/eval_phase/test_stream/Task000",
        "MacroF1": "Top1_MacroF1_Stream/eval_phase/test_stream/Task000",
        "Loss": "Loss_Stream/eval_phase/test_stream/Task000"
        }

        metric_key = metric_key_map.get(self.optimize_metric, "Top1_Acc_Stream/eval_phase/test_stream/Task000")
        score = last_result.get(metric_key, 0.0)

        # Loss minimize edilmek istendiginde isareti �evir
        if self.optimize_metric == "Loss":
             score = -score

        float_results = {k: float(v) for k, v in last_result.items() if isinstance(v, (float, int))}

        trial_result_path = self.results_dir / f"hpo_trial_{trial.number}.json"
        with open(trial_result_path, "w") as f:
            json.dump({
                "params": trial.params,
                "score": score,
                "metrics": float_results
            }, f, indent=2)

        return score

    def run(self):
        print(f"🔍 Starting HPO for strategy: {self.strategy_name}")
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials)

        # Save study summary
        df = study.trials_dataframe()
        df.to_csv(self.results_dir / "hpo_summary.csv", index=False)

        importances = get_param_importances(study)
        with open(self.results_dir / "hpo_param_importance.json", "w") as f:
            json.dump(importances, f, indent=2)
        with open(self.results_dir / "best_params.json", "w") as f:
            json.dump(study.best_trial.params, f, indent=2)

        print("\n✅ HPO Finished. Best trial:")
        print(study.best_trial)
        return study.best_trial.params
    
    
    '''
    from cl_framework.hpo import HyperparameterOptimizer

    hpo = HyperparameterOptimizer(
        benchmark=hpo_benchmark,
        input_size=input_size,
        n_labels=n_labels,
        strategy_name="AGEM",
        dataset_name="amb",
        n_trials=20
    )

    best_params = hpo.run()
    
    '''