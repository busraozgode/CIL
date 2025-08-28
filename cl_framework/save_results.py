import torch
import numpy as np
import pandas as pd
import os
import json


class JSONFormatter:
    """Helper to convert objects to JSON-serializable types."""
    
    @staticmethod
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: JSONFormatter.convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [JSONFormatter.convert(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(JSONFormatter.convert(v) for v in obj)
        else:
            return obj


class ResultSaver:
    """Saves evaluation metrics and results to JSON and CSV files."""
    def __init__(self, eval_plugin, save_dir, filename_prefix, n_experiences=4):
        self.eval_plugin = eval_plugin
        self.prefix = filename_prefix
        self.save_dir = os.path.join(save_dir, filename_prefix)  # Create subfolder per strategy
        os.makedirs(self.save_dir, exist_ok=True)  # Ensure folder exists
        
        self.n_experiences = n_experiences
        self.metric_dict = eval_plugin.get_all_metrics()

    def save_all(self):
        self._save_json()
        df_exp = self._extract_metrics()
        self._save_csv(df_exp)
        #return self.metric_dict, df_exp

    def _save_json(self):
        path = os.path.join(self.save_dir, f"{self.prefix}_metrics.json")
        with open(path, 'w') as f:
            json.dump(JSONFormatter.convert(self.metric_dict), f, indent=4)
        print(f"[✓] Metrics JSON saved to: {path}")

    def _save_csv(self, df):
        path = os.path.join(self.save_dir, f"{self.prefix}_exp_metrics.csv")
        df.to_csv(path, index=False)
        print(f"[✓] Experiment results CSV saved to: {path}")

    def _extract_metrics(self):
        exp_metrics = []
        exp_ids = [f"Exp{str(i).zfill(3)}" for i in range(self.n_experiences)]

        for exp_id in exp_ids:
            exp_metrics.extend(self._collect_metrics_for_experience(exp_id))

        exp_metrics.extend(self._collect_stream_level_metrics())
        return pd.DataFrame(exp_metrics)

    def _collect_metrics_for_experience(self, exp_id):
        acc_key = f'Top1_Acc_Exp/eval_phase/test_stream/Task000/{exp_id}'
        f1_key = f'MacroF1/eval_phase/test_stream/Task000/{exp_id}'
        loss_key = f'Loss_Exp/eval_phase/test_stream/Task000/{exp_id}'
        forgetting_key = f'ExperienceForgetting/eval_phase/test_stream/Task000/{exp_id}'

        if acc_key not in self.metric_dict:
            return []

        epochs = self.metric_dict[acc_key][0]
        accs = self.metric_dict[acc_key][1]
        f1s = self.metric_dict.get(f1_key, [[], [None] * len(accs)])[1]
        losses = self.metric_dict.get(loss_key, [[], [None] * len(accs)])[1]

        forgettings = [None] * len(accs)
        if forgetting_key in self.metric_dict:
            forget_epochs = self.metric_dict[forgetting_key][0]
            forget_values = self.metric_dict[forgetting_key][1]
            forgettings = [forget_values[forget_epochs.index(ep)] if ep in forget_epochs else None for ep in epochs]

        return [
            {
                "Experience": exp_id,
                "Epoch": epoch,
                "Accuracy": acc,
                "Macro_F1": f1,
                "Loss": loss,
                "Forgetting": forgetting
            }
            for epoch, acc, f1, loss, forgetting in zip(epochs, accs, f1s, losses, forgettings)
        ]

    def _collect_stream_level_metrics(self):
        acc_key = 'Top1_Acc_Stream/eval_phase/test_stream/Task000'
        f1_key = 'MacroF1/eval_phase/test_stream/Task000'
        loss_key = 'Loss_Stream/eval_phase/test_stream/Task000'
        forgetting_key = 'StreamForgetting/eval_phase/test_stream'

        if acc_key not in self.metric_dict:
            return []

        epochs = self.metric_dict[acc_key][0]
        accs = self.metric_dict[acc_key][1]
        f1s = self.metric_dict.get(f1_key, [[], [None] * len(accs)])[1]
        losses = self.metric_dict.get(loss_key, [[], [None] * len(accs)])[1]

        forgettings = [None] * len(accs)
        if forgetting_key in self.metric_dict:
            forget_epochs = self.metric_dict[forgetting_key][0]
            forget_values = self.metric_dict[forgetting_key][1]
            forgettings = [forget_values[forget_epochs.index(ep)] if ep in forget_epochs else None for ep in epochs]

        return [
            {
                "Experience": "Stream",
                "Epoch": epoch,
                "Accuracy": acc,
                "Macro_F1": f1,
                "Loss": loss,
                "Forgetting": forgetting
            }
            for epoch, acc, f1, loss, forgetting in zip(epochs, accs, f1s, losses, forgettings)
        ]
