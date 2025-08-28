# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 12:21:07 2025

@author: Busra Ozgode Yigin
"""
import numpy as np
import torch
from torch.utils.data import TensorDataset, ConcatDataset
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks import GenericCLScenario
from sklearn.model_selection import train_test_split
from avalanche.benchmarks import benchmark_from_datasets

# Protocol names
protocols = ['CL', 'DR', 'iD', 'SM2', 'SW', 'X10v2', 'X10v3']

def create_experience(X, y, class_subset, task_label):
    """Filter data by class and wrap in AvalancheDataset"""
    mask = np.isin(y, class_subset)
    X_sel = torch.tensor(X[mask], dtype=torch.float32)
    y_sel = torch.tensor(y[mask], dtype=torch.long)
    td = TensorDataset(X_sel, y_sel)
    aval_dataset = AvalancheDataset(td)

    # Assign task_labels attribute: one label per sample
    aval_dataset.task_labels = [task_label] * len(aval_dataset)

    return aval_dataset

# Experience containers
train_experiences = []
test_datasets = []

for i, protocol in enumerate(protocols):
    # Load data
    X = np.load(f"data/inter-datasets/X_{protocol}.npy")
    y = np.load(f"data/inter-datasets/y_{protocol}.npy")

    # Train/test split (stratify to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train Experiences
    train_experiences.append(create_experience(X_train, y_train, [0, 1], task_label=i))
    train_experiences.append(create_experience(X_train, y_train, [2, 3], task_label=i))

    # Collect all test data (full protocol test set, not split by class)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)
    test_datasets.append(TensorDataset(X_test, y_test))

# Combine all protocol test sets into one unified test set
full_test_dataset = ConcatDataset(test_datasets)
shared_test_set = AvalancheDataset(full_test_dataset)

# Build scenario
benchmark = benchmark_from_datasets(
    train_datasets=train_experiences,  
    test_datasets=[shared_test_set]     
)


for experience in benchmark.train_datasets_stream:
    # experience burada bir DatasetExperience objesi
    print(experience.current_experience)  # index numarası, 0'dan başlayarak artar
    print(len(experience.dataset))        # o experience içindeki örnek sayısı

for test_exp in benchmark.test_datasets_stream:
    print(test_exp.current_experience)
    print(len(test_exp.dataset))