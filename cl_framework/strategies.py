import torch.nn as nn
from avalanche.training import (
    ICaRL, MIR, AGEM, Replay, EWC, LwF, MAS, BiC
)


class StrategyBuilder:
    """Base class for continual learning strategy builders."""
    def __init__(self, model, optimizer, evaluator, device, params):
        self.model = model
        self.optimizer = optimizer
        self.evaluator = evaluator
        self.device = device
        self.params = params

    def build(self):
        raise NotImplementedError("Subclasses must implement this method.")


class ICaRLStrategyBuilder(StrategyBuilder):
    def build(self):
        feature_extractor = nn.Sequential(*list(self.model.net.children())[:-1])
        classifier = self.model.net[-1]
        return ICaRL(
            feature_extractor=feature_extractor,
            classifier=classifier,
            optimizer=self.optimizer,
            memory_size=self.params.get("memory_size", 300),
            buffer_transform=None,
            fixed_memory=True,
            train_mb_size=self.params.get("train_mb_size", 64),
            train_epochs=self.params.get("train_epochs"),
            eval_mb_size=self.params.get("eval_mb_size", 64),
            device=self.device,
            evaluator=self.evaluator
        )


class MIRStrategyBuilder(StrategyBuilder):
    def build(self):
        return MIR(
            model=self.model,
            optimizer=self.optimizer,
            criterion=nn.CrossEntropyLoss(),
            mem_size=self.params.get("memory_size", 300),
            subsample=self.params.get("subsample"),
            train_mb_size=self.params.get("train_mb_size", 64),
            train_epochs=self.params.get("train_epochs"),
            eval_mb_size=self.params.get("eval_mb_size", 64),
            device=self.device,
            evaluator=self.evaluator
        )


class AGEMStrategyBuilder(StrategyBuilder):
    def build(self):
        return AGEM(
            model=self.model,
            optimizer=self.optimizer,
            patterns_per_exp=self.params.get("memory_size", 300),
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=self.params.get("train_mb_size", 64),
            train_epochs=self.params.get("train_epochs"),
            eval_mb_size=self.params.get("eval_mb_size", 64),
            device=self.device,
            evaluator=self.evaluator
        )


class ReplayStrategyBuilder(StrategyBuilder):
    def build(self):
        return Replay(
            model=self.model,
            optimizer=self.optimizer,
            mem_size=self.params.get("memory_size", 300),
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=self.params.get("train_mb_size", 64),
            train_epochs=self.params.get("train_epochs"),
            eval_mb_size=self.params.get("eval_mb_size", 64),
            device=self.device,
            evaluator=self.evaluator
        )


class EWCStrategyBuilder(StrategyBuilder):
    def build(self):
        return EWC(
            model=self.model,
            optimizer=self.optimizer,
            ewc_lambda=self.params.get("ewc_lambda", 0.4),
            mode=self.params.get("ewc_mode", "separate"),  # 'separate' or 'online'
            #decay_factor=self.params.get("decay_factor", 0.1),  # used if mode is 'online'
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=self.params.get("train_mb_size", 64),
            train_epochs=self.params.get("train_epochs"),
            eval_mb_size=self.params.get("eval_mb_size", 64),
            device=self.device,
            evaluator=self.evaluator
        )


class LWFStrategyBuilder(StrategyBuilder):
    def build(self):
        return LwF(
            model=self.model,
            optimizer=self.optimizer,
            alpha=self.params.get("alpha", 1),
            temperature=self.params.get("temperature", 2),
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=self.params.get("train_mb_size", 64),
            train_epochs=self.params.get("train_epochs"),
            eval_mb_size=self.params.get("eval_mb_size", 64),
            device=self.device,
            evaluator=self.evaluator
        )


class MASStrategyBuilder(StrategyBuilder):
    def build(self):
        return MAS(
            model=self.model,
            optimizer=self.optimizer,
            lambda_reg=self.params.get("lambda_reg", 1),
            criterion=nn.CrossEntropyLoss(),
            train_mb_size=self.params.get("train_mb_size", 64),
            train_epochs=self.params.get("train_epochs"),
            eval_mb_size=self.params.get("eval_mb_size", 64),
            device=self.device,
            evaluator=self.evaluator
        )


class BICStrategyBuilder(StrategyBuilder):
    def build(self):
        return BiC(
            model=self.model,
            optimizer=self.optimizer,
            mem_size=self.params.get("memory_size", 300),
            criterion=nn.CrossEntropyLoss(),
            val_percentage = self.params.get("val_percentage", 0.1), # hyperparameter used to set the percentage of exemplars in the val set.
            T=self.params.get("T", 2), # hyperparameter used to set the temperature used in stage 1.
            stage_2_epochs = self.params.get("stage_2_epochs", 200), # hyperparameter used to set the amount of epochs of stage 2.
            lamb = self.params.get("lamb", -1), # hyperparameter used to balance the distilling loss and the classification loss.
            train_mb_size=self.params.get("train_mb_size", 64),
            train_epochs=self.params.get("train_epochs"),
            eval_mb_size=self.params.get("eval_mb_size", 64),
            device=self.device,
            evaluator=self.evaluator,
        )


class StrategyFactory:
    """
    Factory for creating continual learning strategies by name.
    """
    _builders = {
        "icarl": ICaRLStrategyBuilder,
        "mir": MIRStrategyBuilder,
        "agem": AGEMStrategyBuilder,
        "er": ReplayStrategyBuilder,
        "replay": ReplayStrategyBuilder,
        "ewc": EWCStrategyBuilder,
        "lwf": LWFStrategyBuilder,
        "mas": MASStrategyBuilder,
        "bic": BICStrategyBuilder,
    }

    @staticmethod
    def get_strategy(name, model, optimizer, evaluator, device, **params):
        name = name.lower()
        if name not in StrategyFactory._builders:
            raise ValueError(f"Unknown strategy name: {name}")
        builder_cls = StrategyFactory._builders[name]
        return builder_cls(model, optimizer, evaluator, device, params).build()
