# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 11:35:05 2025

@author: Busra Ozgode Yigin
"""

# Experiment configuration

# Mode: choose between "train", "hpo" or "auto"
MODE = "auto"

MAX_PARALLEL_JOBS = 3

# List of strategies to run


STRATEGIES = [
    "icarl",
    "mir",
    "agem",
    "er",
    "ewc",
    "lwf",
    "mas",
    "bic"
]
'''
STRATEGIES = [
    "bic",
]
'''
# Dataset selection

DATASETS = [
    "amb",
    "segerstolpe",
    "zheng68k",
    "zhengsorted",
    "tm",
    "xin",
    "muraro",
    "baron_mouse",
    "baron_human",
    "celseq2_5cl",
    "10x_5cl"
    ]
'''
DATASETS = [
    "segerstolpe"]
'''
N_TRIALS = 20  # Number of HPO trials

# Common parameters for all strategies
COMMON_PARAMS = {
    "train_epochs": {"type": "int", "low": 3, "high": 10},
    "train_mb_size": 64,
    "eval_mb_size": 64
}

# Strategy-specific parameters (in addition to COMMON_PARAMS)
STRATEGY_PARAMS = {
    "icarl": {
        "memory_size": 300
    },
    "mir": {
        "memory_size": 300,
        "subsample": {"type": "int", "low": 10, "high": 100}
    },
    "agem": {
        "patterns_per_exp": 300,  
    },
    "er": {
        "memory_size": 300
    },
    "ewc": {
        "ewc_lambda": {"type": "float", "low": 1, "high": 1000, "log": True},
        "ewc_mode": "separate",  # can be "separate" or "online" default:separate
        #"decay_factor": {"type": "float", "low": 0.0, "high": 1.0}
    },
    "lwf": {
        "alpha": {"type": "float", "low": 0.5, "high": 0.8},
        "temperature": {"type": "float", "low": 1.0, "high": 5.0}
    },
    "mas": {
        "lambda_reg": {"type": "float", "low": 1, "high": 1000, "log": True}
    },
    "bic": {
        "memory_size": 300,
        "val_percentage": {"type": "float", "low": 0.05, "high": 0.1},
        "T": {"type": "int", "low": 1, "high": 5}, # hyperparameter used to set the temperature used in stage 1.
        "stage_2_epochs": 200, # hyperparameter used to set the amount of epochs of stage 2.
        "lamb": -1 # hyperparameter used to balance the distilling loss and the classification loss.
    }
}


def get_strategy_params(name: str, only_strategy_specific: bool = True):
    """
    Returns the parameters for a given strategy.

    If only_strategy_specific=True, returns only strategy-specific params.
    Otherwise, merges with common params (strategy-specific take precedence).
    """
    strategy_specific = STRATEGY_PARAMS.get(name.lower(), {})
    if only_strategy_specific:
        return strategy_specific
    return {**COMMON_PARAMS, **strategy_specific}


def get_hpo_param_defs(strategy_name: str):
    """
    Merge and return only HPO-eligible params from COMMON and STRATEGY settings.
    """
    strategy_specific = STRATEGY_PARAMS.get(strategy_name.lower(), {})
    merged = {**COMMON_PARAMS, **strategy_specific}

    hpo_params = {
        k: v for k, v in merged.items()
        if isinstance(v, dict) and "type" in v  # only params with HPO ranges
    }
    return hpo_params
