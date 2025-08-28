import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torch.utils.data import TensorDataset
from avalanche.benchmarks import nc_benchmark
import rpy2.robjects as robjects


class DatasetLoader:
    def __init__(self, dataset_name: str):
        self.project_root = Path(__file__).parent.parent
        self.dataset_name = dataset_name.lower()
        self.bench_path = self.project_root / 'data' / self.dataset_name
        self.input_size = None
        self.n_labels = None
        self.benchmark = None
        self.hpo_benchmark = None
        self.n_experiences = None

        self.data_name, self.label_name, self.rdata_name = self._get_filenames()

    def _get_filenames(self):
        if self.dataset_name == 'amb':
            return (
                "Filtered_mouse_allen_brain_data.csv",
                "Labels.csv",
                "CV_folds_3rd_level.RData"
            )
        elif self.dataset_name == 'segerstolpe':
            return (
                "Filtered_Segerstolpe_HumanPancreas_data.csv",
                "Labels.csv",
                "CV_folds.RData"
            )
        elif self.dataset_name == 'celseq2_5cl':
            return (
                "CelSeq2_5cl_data.csv",
                "Labels.csv",
                "CV_folds.RData"
            )
        elif self.dataset_name == '10x_5cl':
            return (
                "10x_5cl_data.csv",
                "Labels.csv",
                "CV_folds.RData"
            )
        elif self.dataset_name == 'zheng68k':
            return (
                "Filtered_68K_PBMC_data.csv",
                "Labels.csv",
                "CV_folds.RData"
            )
        elif self.dataset_name == 'zhengsorted':
            return (
                "Filtered_DownSampled_SortedPBMC_data.csv",
                "Labels.csv",
                "CV_folds.RData"
            )
        elif self.dataset_name == 'tm':
            return (
                "Filtered_TM_data.csv",
                "Labels.csv",
                "CV_folds.RData"
            )
        elif self.dataset_name == 'xin':
            return (
                "Filtered_Xin_HumanPancreas_data.csv",
                "Labels.csv",
                "CV_folds.RData"
            )
        elif self.dataset_name == 'muraro':
            return (
                "Filtered_Muraro_HumanPancreas_data.csv",
                "Labels.csv",
                "CV_folds.RData"
            )
        elif self.dataset_name == 'baron_mouse':
            return (
                "Filtered_MousePancreas_data.csv",
                "Labels.csv",
                "CV_folds.RData"
            ) 
        elif self.dataset_name == 'baron_human':
            return (
                "Filtered_Baron_HumanPancreas_data.csv",
                "Labels.csv",
                "CV_folds.RData"
            )
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def _load_rdata_split_info(self, rdata_path):
        robjects.r['load'](str(rdata_path))
        cells_to_keep = np.array(robjects.r['Cells_to_Keep'], dtype=bool)
        col_index = np.array(robjects.r['col_Index'], dtype=int) - 1
        return cells_to_keep, col_index

    def _load_and_filter_data(self, data_path, label_path, cells_to_keep):
        data_df = pd.read_csv(data_path, index_col=0,sep=',')
        label_df = pd.read_csv(label_path, header=0,
                                  index_col=None, sep=',')
        
        return data_df.iloc[cells_to_keep], label_df.iloc[cells_to_keep]

    def _preprocess_data(self, data_df, label_df):
        X = np.log1p(data_df.to_numpy())

        if label_df.shape[1] > 1:
            labels = label_df.iloc[:, -2].to_numpy()
        else:
            labels = label_df.to_numpy().ravel()

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(labels)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def _split_data(self, X, y):
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42)

        datasets = {
            "train_full": TensorDataset(X_train_full, y_train_full),
            "train": TensorDataset(X_train, y_train),
            "val": TensorDataset(X_val, y_val),
            "test": TensorDataset(X_test, y_test)
        }

        return datasets, X_train.shape[1], len(torch.unique(y_train))

    def _create_benchmarks(self, datasets, n_labels):
        n_experiences = int(np.floor(n_labels / 2))
        print('number of labels: {n_labels}, n_experiences : {n_experiences}')
        is_divisible = n_labels % n_experiences == 0

        if not is_divisible:
            print(f"{n_labels} is NOT divisible by {n_experiences}")
            f_exp = int(np.floor(n_labels / n_experiences) + n_labels - np.floor(n_labels / n_experiences) * n_experiences)
            per_exp = {0: f_exp}
            hpo_benchmark = nc_benchmark(
                train_dataset=datasets["train"],
                test_dataset=datasets["val"],
                n_experiences=n_experiences,
                task_labels=False,
                shuffle=True,
                per_exp_classes=per_exp,
                seed=42
            )
            benchmark = nc_benchmark(
                train_dataset=datasets["train_full"],
                test_dataset=datasets["test"],
                n_experiences=n_experiences,
                task_labels=False,
                shuffle=True,
                per_exp_classes=per_exp,
                seed=42
            )
        else:
            print(f"{n_labels} is divisible by {n_experiences}")
            hpo_benchmark = nc_benchmark(
                train_dataset=datasets["train"],
                test_dataset=datasets["val"],
                n_experiences=n_experiences,
                task_labels=False,
                shuffle=True,
                seed=42
            )
            benchmark = nc_benchmark(
                train_dataset=datasets["train_full"],
                test_dataset=datasets["test"],
                n_experiences=n_experiences,
                task_labels=False,
                shuffle=True,
                seed=42
            )

        return benchmark, hpo_benchmark, n_experiences

    def load(self):
        data_path = self.bench_path / self.data_name
        label_path = self.bench_path / self.label_name
        rdata_path = self.bench_path / self.rdata_name

        print(f"Loading RData file: {rdata_path}")
        cells_to_keep, _ = self._load_rdata_split_info(rdata_path)

        data_df, label_df = self._load_and_filter_data(data_path, label_path, cells_to_keep)
        X, y = self._preprocess_data(data_df, label_df)

        print(f"Shape of label array: {y.shape}")
        unique, counts = torch.unique(y, return_counts=True)
        print("Classes of the dataset:")
        for u, c in zip(unique.tolist(), counts.tolist()):
            print(f"Class {u}: {c} samples")

        datasets, self.input_size, self.n_labels = self._split_data(X, y)
        self.benchmark, self.hpo_benchmark, self.n_experiences = self._create_benchmarks(datasets, self.n_labels)
        self.X = X
        self.y = y
        return self
    
    #For the Server to load tensors from the saved files
    def load_from_tensor(self):
        data_path = self.bench_path / "X.pt"
        label_path = self.bench_path / "y.pt"

        print(f"Loading tensors:\n  X: {data_path}\n  y: {label_path}")
        X = torch.load(data_path)
        y = torch.load(label_path)

        print(f"Shape of label array: {y.shape}")
        unique, counts = torch.unique(y, return_counts=True)
        print("Classes of the dataset:")
        for u, c in zip(unique.tolist(), counts.tolist()):
            print(f"Class {u}: {c} samples")

        datasets, self.input_size, self.n_labels = self._split_data(X, y)
        self.benchmark, self.hpo_benchmark, self.n_experiences = self._create_benchmarks(datasets, self.n_labels)

        self.X = X
        self.y = y

        return self

    def to_tuple(self):
        return self.benchmark, self.hpo_benchmark, self.input_size, self.n_labels, self.n_experiences
    
    '''
    loader = DatasetLoader("amb").load()

    benchmark = loader.benchmark
    hpo_benchmark = loader.hpo_benchmark
    input_size = loader.input_size
    n_labels = loader.n_labels
    '''