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
from avalanche.benchmarks.utils.data_attribute import DataAttribute
#from avalanche.training.plugins import Experience
from collections import defaultdict
from torch.utils.data import Subset

class InterDatasetBenchmarkBuilder:
    def __init__(self, data_dir="data/inter-datasets", protocols=None):
        if protocols is None:
            protocols = ['CL', 'DR', 'iD', 'SM2', 'SW', 'X10v2', 'X10v3']
        self.protocols = protocols
        self.data_dir = data_dir
        self.input_size = None
        self.n_labels = None
        self.benchmark = None
        self.hpo_benchmark = None
        self.n_experiences = None

    def add_task_set(self, av_dataset):
        task_set = defaultdict(list)
        for i, task_label in enumerate(av_dataset.targets_task_labels[:]):
            task_set[int(task_label)].append(i)
            av_dataset.task_set = {
                task_id: Subset(av_dataset, indices)
                for task_id, indices in task_set.items()
                }    
    def create_experience(self, X, y, class_subset, task_label):
        mask = np.isin(y, class_subset)
        X_sel = torch.tensor(X[mask], dtype=torch.float32)
        y_sel = torch.tensor(y[mask], dtype=torch.long)
        td = TensorDataset(X_sel, y_sel)
        aval_dataset = AvalancheDataset(td)
        aval_dataset.targets = y_sel
        
        task_labels_tensor = torch.zeros(len(aval_dataset), dtype=torch.long)
        aval_dataset.task_labels = DataAttribute(
            name="task_labels",
            data=task_labels_tensor
            )
        
        # targets_task_labels 
        aval_dataset.targets_task_labels = DataAttribute(
            name="targets_task_labels",
            data=task_labels_tensor
            )
        
        self.add_task_set(aval_dataset)
        
        return aval_dataset
    
    def _split_data(self, X, y):
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

        datasets = {
            "X_train_full": X_train_full, 
            "y_train_full": y_train_full,
            "X_train": X_train, 
            "y_train": y_train,
            "X_val": X_val, 
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test
        }

        return datasets, X_train.shape[1], len(np.unique(y_train))

    def finalize_benchmark(self, benchmark, n_labels, n_experiences, input_size):
        benchmark.n_classes = n_labels
        benchmark.n_experiences = n_experiences
        benchmark.n_tasks = 1  
        benchmark.input_shape = input_size

        
        if hasattr(benchmark, "train_datasets_stream"):
            benchmark.train_stream = benchmark.train_datasets_stream
        if hasattr(benchmark, "test_datasets_stream"):
            benchmark.test_stream = benchmark.test_datasets_stream

       
        for exp in benchmark.train_stream:
            exp.benchmark = benchmark
        for exp in benchmark.test_stream:
            exp.benchmark = benchmark

        # calculate n_classes_per_exp 
        n_classes_per_exp = []
        all_classes = set()
        for exp in benchmark.train_stream:
            targets = exp.dataset.targets if hasattr(exp.dataset, "targets") else [y for _, y in exp.dataset]
            classes = set(targets)
            n_classes_per_exp.append(len(classes))
            all_classes.update(classes)

        benchmark.n_classes_per_exp = n_classes_per_exp
        benchmark.classes_order = sorted(list(all_classes))
    def enrich_experiences_with_class_info(self, experiences):
        for experience in experiences:
            dataset = experience.dataset
            targets = dataset.targets if hasattr(dataset, "targets") else [y for _, y in dataset]
            classes = sorted(set(int(t) for t in targets))
            experience.classes_in_this_experience = classes
    def build(self):
        train_experiences = []
        train_full_experiences = []
        val_datasets = []
        test_datasets = []

        for i, protocol in enumerate(self.protocols):
            X = np.load(f"{self.data_dir}/X_{protocol}.npy")
            y = np.load(f"{self.data_dir}/y_{protocol}.npy")

            datasets, self.input_size, self.n_labels = self._split_data(X, y)

            train_full_experiences.append(self.create_experience(datasets["X_train_full"],datasets["y_train_full"], [0, 1], task_label=i))
            train_full_experiences.append(self.create_experience(datasets["X_train_full"],datasets["y_train_full"], [2, 3], task_label=i))
            self.n_experiences = len(train_full_experiences)
            train_experiences.append(self.create_experience(datasets["X_train"],datasets["y_train"], [0, 1], task_label=i))
            train_experiences.append(self.create_experience(datasets["X_train"],datasets["y_train"], [2, 3], task_label=i))
        
            X_test = torch.tensor(datasets["X_test"], dtype=torch.float32)
            y_test = torch.tensor(datasets["y_test"], dtype=torch.long)
            test_datasets.append(TensorDataset(X_test, y_test))
            
            # Val For HPO
            X_val = torch.tensor(datasets["X_val"], dtype=torch.float32)
            y_val = torch.tensor(datasets["y_val"], dtype=torch.long)
            val_datasets.append(TensorDataset(X_val, y_val))
        

        full_test_dataset = ConcatDataset(test_datasets)
        shared_test_set = AvalancheDataset(full_test_dataset)
        shared_test_set.targets = datasets["y_test"]
        full_val_dataset = ConcatDataset(val_datasets)
        shared_val_set = AvalancheDataset(full_val_dataset)
        shared_val_set.targets = datasets["y_val"]
        
        
        
        self.benchmark = benchmark_from_datasets(
            train_datasets=train_full_experiences,
            test_datasets=[shared_test_set]
        )
        
        self.hpo_benchmark = benchmark_from_datasets(
            train_datasets=train_experiences,
            test_datasets=[shared_val_set]
        )
        
        self.finalize_benchmark(self.benchmark, self.n_labels, self.n_experiences, self.input_size)
        self.finalize_benchmark(self.hpo_benchmark, self.n_labels, self.n_experiences, self.input_size)
        self.enrich_experiences_with_class_info(self.benchmark.train_stream)
        self.enrich_experiences_with_class_info(self.benchmark.test_stream)
        self.enrich_experiences_with_class_info(self.hpo_benchmark.train_stream)
        self.enrich_experiences_with_class_info(self.hpo_benchmark.test_stream)
        
        return self.benchmark, self.hpo_benchmark, self.input_size, self.n_labels, self.n_experiences