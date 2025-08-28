# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 13:24:25 2025

@author: Admin
"""
from avalanche.benchmarks import benchmark_from_datasets
from avalanche.benchmarks.utils import AvalancheDataset
from torch.utils.data import TensorDataset, ConcatDataset
import torch
import numpy as np
from sklearn.model_selection import train_test_split

class InterDatasetBenchmarkBuilder:
    def __init__(self, data_dir="data/inter-datasets", protocols=None):
        if protocols is None:
            protocols = ['CL', 'DR', 'iD', 'SM2', 'SW', 'X10v2', 'X10v3']
        self.protocols = protocols
        self.data_dir = data_dir

    def create_experience(self, X, y, class_subset, task_label):
        mask = np.isin(y, class_subset)
        X_sel = torch.tensor(X[mask], dtype=torch.float32)
        y_sel = torch.tensor(y[mask], dtype=torch.long)
        td = TensorDataset(X_sel, y_sel)
        aval_dataset = AvalancheDataset(td)
        aval_dataset.task_labels = [task_label] * len(aval_dataset)
        return aval_dataset

    def build(self):
        train_experiences = []
        test_datasets = []

        for i, protocol in enumerate(self.protocols):
            X = np.load(f"{self.data_dir}/X_{protocol}.npy")
            y = np.load(f"{self.data_dir}/y_{protocol}.npy")

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42
            )

            train_experiences.append(self.create_experience(X_train, y_train, [0, 1], task_label=i))
            train_experiences.append(self.create_experience(X_train, y_train, [2, 3], task_label=i))

            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.long)
            test_datasets.append(TensorDataset(X_test, y_test))

        full_test_dataset = ConcatDataset(test_datasets)
        shared_test_set = AvalancheDataset(full_test_dataset)

        benchmark = benchmark_from_datasets(
            train_datasets=train_experiences,
            test_datasets=[shared_test_set]
        )

        return benchmark