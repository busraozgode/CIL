# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 14:01:07 2025

@author: Admin
"""
from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metric_utils import get_metric_name
from sklearn.metrics import f1_score
import torch


class MacroF1PluginMetric(PluginMetric[float]):
    def __init__(self, eval_phase=True, epoch=False, experience=False, stream=False):
        super().__init__()

        self.eval_phase = eval_phase
        self.emit_at_epoch = epoch
        self.emit_at_experience = experience
        self.emit_at_stream = stream

        self.all_preds = []
        self.all_targets = []

    def reset(self, **kwargs):
        self.all_preds = []
        self.all_targets = []

    def update(self, strategy):
        preds = torch.argmax(strategy.mb_output, dim=1).cpu().numpy()
        targets = strategy.mb_y.cpu().numpy()
        self.all_preds.extend(preds)
        self.all_targets.extend(targets)

    def result(self):
        if len(self.all_targets) == 0:
            return 0.0
        return f1_score(self.all_targets, self.all_preds, average="macro")

    def after_eval_iteration(self, strategy):
        if self.eval_phase:
            self.update(strategy)

    def after_training_iteration(self, strategy):
        if not self.eval_phase:
            self.update(strategy)

    def before_eval(self, strategy, **kwargs):
        if self.eval_phase:
            self.reset()

    def before_training(self, strategy, **kwargs):
        if not self.eval_phase:
            self.reset()

    def after_eval_epoch(self, strategy):
        if self.eval_phase and self.emit_at_epoch:
            return self._package_result(strategy)

    def after_training_epoch(self, strategy):
        if not self.eval_phase and self.emit_at_epoch:
            return self._package_result(strategy)

    def after_eval_exp(self, strategy):
        if self.eval_phase and self.emit_at_experience:
            return self._package_result(strategy, add_experience=True)

    def after_training_exp(self, strategy):
        if not self.eval_phase and self.emit_at_experience:
            return self._package_result(strategy)

    def after_eval(self, strategy):
        if self.eval_phase and self.emit_at_stream:
            return self._package_result(strategy, add_experience=False)

    def after_training(self, strategy):
        if not self.eval_phase and self.emit_at_stream:
            return self._package_result(strategy)

    def _package_result(self, strategy, add_experience=True, add_task=True):
        metric_value = self.result()
        metric_name = get_metric_name(self, strategy, add_experience=add_experience, add_task=add_task)
        plot_x_position = strategy.clock.train_iterations
        return [MetricValue(self, metric_name, metric_value, plot_x_position)]

    def __str__(self):
        return "MacroF1"