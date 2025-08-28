# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 12:07:20 2025
@author: Busra Ozgode Yigin
"""

import torch
from pathlib import Path
import os
from avalanche.training.plugins import EvaluationPlugin
from avalanche.logging import InteractiveLogger, TensorboardLogger
from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics, forgetting_metrics,
    timing_metrics, confusion_matrix_metrics
)

from cl_framework.strategies import StrategyFactory
from cl_framework.StandaloneMetrics import MacroF1PluginMetric
from cl_framework.simple_mlp import MLPFactory
from cl_framework.save_results import ResultSaver


class ContinualLearningTrainer:
    def __init__(self, benchmark, input_size, n_labels, n_experiences,
                 strategy_name, dataset_name, best_params=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark = benchmark
        self.input_size = input_size
        self.n_labels = n_labels
        self.n_experiences = n_experiences
        self.strategy_name = strategy_name
        self.dataset_name = dataset_name
        self.best_params = best_params or {}
        
        self.model = MLPFactory().create(input_size=self.input_size, num_classes=self.n_labels)
        self.model = self.model.to(self.device)

        lr = self.best_params.get("learning_rate", 0.00033)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.eval_plugin = EvaluationPlugin(
            accuracy_metrics(epoch=True, experience=True, stream=True),
            loss_metrics(epoch=True, experience=True, stream=True),
            timing_metrics(epoch=True),
            forgetting_metrics(experience=True, stream=True),
            MacroF1PluginMetric(eval_phase=True, epoch=False, experience=True, stream=True),
            confusion_matrix_metrics(
                num_classes=benchmark.n_classes, save_image=False, stream=True
            ),
            loggers=[InteractiveLogger(), TensorboardLogger()]
        )

        self.strategy = StrategyFactory.get_strategy(
            self.strategy_name,self. model, self.optimizer, self.eval_plugin, self.device, **self.best_params
        )
            

    def train(self):
        print(self.strategy)
        print(f"\n🎯 Training full stream with best params: {self.best_params}")
        results = []

        for experience in self.benchmark.train_stream:
            labels_in_exp = set([y for _, y, *_ in experience.dataset])
            print(f"\nTraining on experience {experience.current_experience} with labels: {sorted(labels_in_exp)}")
            self.strategy.train(experience)
            print("Training completed")

            print("Computing accuracy on the whole test set")
            results.append(self.strategy.eval(self.benchmark.test_stream))

        self.save_results()
        return results

    def save_results(self):
        project_root = Path(__file__).parent.parent
        data_dir = project_root / 'Results'
        bench_path = data_dir / self.dataset_name.lower()
        os.makedirs(bench_path, exist_ok=True)

        saver = ResultSaver(
            eval_plugin=self.eval_plugin,
            save_dir=bench_path,
            filename_prefix=self.strategy_name,
            n_experiences=self.n_experiences
        )
    
        #metric_dict, df_exp = saver.save_all()
        saver.save_all()
        print(f"✅ Results saved to: {bench_path}")


# Example usage (you'd call this from main.py or another script):
# trainer = ContinualLearningTrainer(benchmark, input_size, n_labels, n_experiences, "AGEM", "amb")
# trainer.train()