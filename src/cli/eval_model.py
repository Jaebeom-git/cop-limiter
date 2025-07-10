import argparse
import torch
import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt

from cli.train import TrainCommand
from cli.config_manager import ConfigManager

from data.AddBiomechanicsDataset import AddBiomechanicsDataset
from loss.RegressionLossEvaluator import RegressionLossEvaluator
from models.model_selector import select_model

from torch.utils.data import DataLoader
from torch.amp import autocast

class EvaluateCommand(TrainCommand):
    """
    EvaluateCommand inherits from TrainCommand to reuse functionality like 
    get_dataloader_evaluator(), skeleton loading, etc.

    Steps:
    1) --config-dir <some/folder> -> gather all config files
    2) The first config will be used to create dev/test DataLoaders
    3) For each config, we load best dev checkpoint, evaluate on dev/test,
       collecting dev_force, dev_cop, dev_moment, dev_tau, etc.
    4) We gather df_dev, df_test, and finally export them to a single Excel file 
       with two sheets (or as you prefer).
    """
    def __init__(self):
        super().__init__()
        self.df_dev = None
        self.df_test = None

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('evaluate', help='Evaluate models in a config directory on dev/test sets')

        subparser.add_argument('--dataset-home', type=str, default='../data',
                               help='Path to the dataset root (with dev/test subfolders).')
        subparser.add_argument('--data-loading-workers', type=int, default=8,
                               help='Number of separate processes for data loading.')
        subparser.add_argument('--geometry-folder', type=str, default=None,
                               help='Path to the bone mesh geometry folder.')
        subparser.add_argument('--config-dir', type=str, required=True,
                               help='Directory containing multiple config YAML files to evaluate.')
        subparser.add_argument('--result-path', type=str, default='eval_results.xlsx',
                               help='Path to save the combined results (with two sheets: dev/test).')

        subparser.add_argument('--predict-grf-components', type=int, nargs='+', default=[i for i in range(6)])
        subparser.add_argument('--predict-cop-components', type=int, nargs='+', default=[i for i in range(6)])
        subparser.add_argument('--predict-moment-components', type=int, nargs='+', default=[i for i in range(6)])
        subparser.add_argument('--predict-wrench-components', type=int, nargs='+', default=[i for i in range(12)])
        
    def run(self, args: argparse.Namespace) -> bool:
        if 'command' in args and args.command != 'evaluate':
            return False

        self.set_multiprocessing_start_method()

        # 1) Gather all config files in config_dir
        config_dir = args.config_dir
        # For example, we might be using .yaml or .json. Let's say .yaml here:
        config_paths = sorted(glob.glob(os.path.join(config_dir, "*.yaml")))
        if not config_paths:
            logging.error(f"No .yaml files found in {config_dir}")
            return False
        
        # 2) The FIRST config -> build dev/test DataLoader
        first_config_path = config_paths[0]
        print(f"[EvaluateCommand] First config for data loader: {first_config_path}")
        config_manager = ConfigManager(first_config_path)
        self.config = config_manager.config
        device = torch.device(self.config.device)
        geometry = self.ensure_geometry(args.geometry_folder)

        dev_path = os.path.join(args.dataset_home, 'dev')
        test_path = os.path.join(args.dataset_home, 'test')

        dev_dataloader, dev_loss_eval = self.get_dataloader_evaluator(
            dataset_path=dev_path,
            split='dev',
            config=self.config,
            geometry=geometry,
            device=device,
            data_loading_workers=args.data_loading_workers
        )
        test_dataloader, test_loss_eval = None, None
        if os.path.isdir(test_path):
            test_dataloader, test_loss_eval = self.get_dataloader_evaluator(
                dataset_path=test_path,
                split='test',
                config=self.config,
                geometry=geometry,
                device=device,
                data_loading_workers=args.data_loading_workers
            )

        # Prepare lists to store results: dev_data, test_data
        dev_data = []
        test_data = []

        # 3) For each config, create a model, load best dev checkpoint, evaluate
        for cfg_path in config_paths:
            print("\n===========================================")
            print(f"[EvaluateCommand] Evaluating config: {cfg_path}")
            print("===========================================")

            # Load config
            cfg_manager = ConfigManager(cfg_path)
            local_cfg = cfg_manager.config
            device = torch.device(local_cfg.device)
            geometry = self.ensure_geometry(args.geometry_folder)

            # Model name
            model_name = local_cfg.model_name
            # Create model
            model = select_model(local_cfg.model_params, device)
            # Load best dev checkpoint
            checkpoint_dir = os.path.join(local_cfg.checkpoint_dir, local_cfg.model_name)
            self.load_best_model(model, checkpoint_dir, opt='model', model_type='dev')
            model.eval()

            # Evaluate dev
            dev_force, dev_cop, dev_moment, dev_tau = (None,)*4
            if dev_dataloader is not None:
                print(f"Evaluating model: Validation set")
                dev_force, dev_cop, dev_moment, dev_tau, dev_tau_std = self.evaluate_model(args, model, dev_dataloader, dev_loss_eval, local_cfg.device)

            # Evaluate test
            test_force, test_cop, test_moment, test_tau = (None,)*4
            if test_dataloader is not None:
                print(f"Evaluating model: Test set")
                test_force, test_cop, test_moment, test_tau, test_tau_std = self.evaluate_model(args, model, test_dataloader, test_loss_eval, local_cfg.device)

            # Record results for dev
            dev_data.append({
                "model": model_name,
                "CoP": dev_cop,
                "Force": dev_force,
                "Moment": dev_moment,
                "Tau": dev_tau,
                "Tau_std": dev_tau_std
            })
            # Record results for test
            test_data.append({
                "model": model_name,
                "CoP": test_cop,
                "Force": test_force,
                "Moment": test_moment,
                "Tau": test_tau,
                "Tau_std": test_tau_std
            })

        # 4) Build DataFrame for dev/test
        df_dev = pd.DataFrame(dev_data)
        df_test = pd.DataFrame(test_data)

        # Example custom sorting
        df_dev["family"] = df_dev["model"].apply(lambda x: x.split('_')[0])
        df_dev["variant"] = df_dev["model"].apply(lambda x: '_'.join(x.split('_')[1:]) if '_' in x else "")
        variant_order = {"BL": 0, "CL": 1, "CL_WL": 2}
        df_dev["variant_order"] = df_dev["variant"].apply(lambda v: variant_order.get(v, 99))
        df_dev = df_dev.sort_values(by=["family", "variant_order"], ascending=[True, True])
        ordered_models = df_dev["model"].tolist()

        df_test = df_test.set_index("model").loc[ordered_models].reset_index()

        # 5) Print results
        print("[Dev DataFrame]")
        print(df_dev[["model", "CoP", "Force", "Moment", "Tau", "Tau_std"]].round(3))
        print("\n[Test DataFrame]")
        print(df_test[["model", "CoP", "Force", "Moment", "Tau", "Tau_std"]].round(3))

        # Store
        self.df_dev = df_dev[["model", "CoP", "Force", "Moment", "Tau", "Tau_std"]]
        self.df_test = df_test[["model", "CoP", "Force", "Moment", "Tau", "Tau_std"]]

        # 6) Save them to a single Excel file with 2 sheets
        # user wants them in "one csv"? Typically CSV doesn't have sheets, so let's do Excel:
        result_path = getattr(args, 'result_path', 'eval_results.xlsx')
        with pd.ExcelWriter(result_path) as writer:
            self.df_dev.to_excel(writer, index=False, sheet_name='dev')
            self.df_test.to_excel(writer, index=False, sheet_name='test')

        print(f"[EvaluateCommand] Saved results to {result_path} (two sheets: dev, test).")

        return True

    def evaluate_model(self, args, model, dataloader, loss_evaluator, device):
        """
        Evaluate the model with the given dataloader, returning Force, CoP, Moment, Tau metrics.
        """
        model.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating", dynamic_ncols=True):
                inputs, labels, batch_subject_indices, batch_trial_indices = batch
                if self.config.amp:
                    with autocast(device_type=device, dtype=torch.bfloat16):
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)

                loss_evaluator(
                    inputs,
                    outputs,
                    labels,
                    batch_subject_indices,
                    batch_trial_indices,
                    args=args,
                    compute_report=True,
                    log_reports_to_wandb=False
                )
        force_err   = np.mean(loss_evaluator.force_reported_metrics)
        cop_err     = np.mean(loss_evaluator.cop_reported_metrics)
        moment_err  = np.mean(loss_evaluator.moment_reported_metrics)
        tau_err     = np.mean(loss_evaluator.tau_reported_metrics)
        tau_err_std = np.mean(loss_evaluator.tau_reported_metrics_std)

        loss_evaluator.print_report(reset=True, args=args)
        return force_err, cop_err, moment_err, tau_err, tau_err_std



