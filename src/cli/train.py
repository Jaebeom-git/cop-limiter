import argparse
import datetime
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, InputDataKeys, OutputDataKeys
from loss.RegressionLossEvaluator import RegressionLossEvaluator
from typing import Dict, Tuple, List
from cli.abstract_command import AbstractCommand
import os
import time
import wandb
import numpy as np
import logging
import subprocess
from cli.config_manager import ConfigManager
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from models.model_selector import select_model
import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="megablocks")
from megablocks.layers.moe import batched_load_balancing_loss, clear_load_balancing_loss, batched_router_zloss, clear_router_zloss
from schedulefree import SGDScheduleFree, AdamWScheduleFree, RAdamScheduleFree
import torch.optim
from typing import Any
from transformers import set_seed, SchedulerType, get_scheduler
import fnmatch

class TrainCommand(AbstractCommand):
    def __init__(self):
        super().__init__()

    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        subparser = subparsers.add_parser('train', help='Train a model on the AddBiomechanics dataset')

        subparser.add_argument('--dataset-home', type=str, default='../data',help='The path to the AddBiomechanics dataset.')
        subparser.add_argument('--data-loading-workers', type=int, default=3, help='Number of separate processes to spawn to load data in parallel.')
        subparser.add_argument('--no-wandb', action='store_true', default=False,help='Log this run to Weights and Biases.')
        subparser.add_argument('--geometry-folder', type=str, default=None, help='Path to the Geometry folder with bone mesh data.')
        subparser.add_argument('--config-path', type=str, required=True, help='Path to the configuration file for model and training settings.')

        subparser.add_argument('--predict-grf-components', type=int, nargs='+', default=[i for i in range(6)], help='Which grf components to train.')
        subparser.add_argument('--predict-cop-components', type=int, nargs='+', default=[i for i in range(6)], help='Which cop components to train.')
        subparser.add_argument('--predict-moment-components', type=int, nargs='+', default=[i for i in range(6)], help='Which moment components to train.')
        subparser.add_argument('--predict-wrench-components', type=int, nargs='+', default=[i for i in range(12)], help='Which wrench components to train.')
        subparser.add_argument('--trial-filter', type=str, nargs='+', default=[""], help='What kind of trials to train/test on.')
        
    def run(self, args: argparse.Namespace):
        if 'command' in args and args.command != 'train':
            return False
        
        # Set the multiprocessing start method
        self.set_multiprocessing_start_method()

        # Load configuration settings
        config_manager = ConfigManager(args.config_path)
        config = config_manager.config
        device = torch.device(config.device)
        epochs = config.epochs
        optimizer_params = config.optimizer_params
        model_type = config.model_name
        checkpoint_dir = os.path.join(os.path.abspath(config.checkpoint_dir), model_type)
        log_to_wandb = not args.no_wandb
        geometry = self.ensure_geometry(args.geometry_folder)
        set_seed(config.seed)

        # Load the data loaders and loss evaluators for training and validation datasets
        logging.info('## Loading datasets with skeletons:')
        train_dataloader, train_loss_evaluator = self.get_dataloader_evaluator(
            dataset_path=os.path.join(args.dataset_home, 'train'),
            split='train',
            config=config,
            geometry=geometry,
            device=device,
            data_loading_workers=args.data_loading_workers
        )
        total_training_steps = epochs * len(train_dataloader)

        dev_dataloader, dev_loss_evaluator = self.get_dataloader_evaluator(
            dataset_path=os.path.join(args.dataset_home, 'dev'),
            split='dev',
            config=config,
            geometry=geometry,
            device=device,
            data_loading_workers=args.data_loading_workers
        )

        ## Test set
        test_path = os.path.join(args.dataset_home, 'test')
        test_flag = os.path.isdir(test_path)
        if test_flag:
            test_dataloader, test_loss_evaluator = self.get_dataloader_evaluator(
                dataset_path=os.path.join(args.dataset_home, 'test'),
                split='test',
                config=config,
                geometry=geometry,
                device=device,
                data_loading_workers=args.data_loading_workers
            )

        # # Update the config with the input and output sizes
        # config_manager.update(num_dofs=train_dataset.num_dofs,
        #                       num_joints=train_dataset.num_joints)

        # Create an instance of the model
        model = select_model(config.model_params, device)
        self.print_model_summary(model)
        # for name, param in model.named_parameters():
        #     print(name)
            
        # Initialize the optimizer
        optimizer_params.warmup_steps = int(getattr(optimizer_params, 'warmup_step_ratio', 5)/100 * total_training_steps)
        optimizer = get_optimizer(model, optimizer_params)
        if "schedulefree" in optimizer_params.optimizer.lower():
            config.SchedulerFree = True
        else:
            config.SchedulerFree = False
        
        # Initialize the scheduler
        if config.SchedulerFree or getattr(config, 'scheduler_params', None) is None:
            scheduler = get_hf_scheduler(optimizer, False, total_training_steps)
        else:
            scheduler = get_hf_scheduler(optimizer, config.scheduler_params, total_training_steps)

        # Load the latest checkpoint if available
        best_train_loss = self.load_best_model(model, checkpoint_dir, opt='loss', model_type='train')
        best_dev_loss = self.load_best_model(model, checkpoint_dir, opt='loss', model_type='dev')
        best_test_loss = self.load_best_model(model, checkpoint_dir, opt='loss', model_type='test')
        start_epoch, wandb_run_id = self.load_latest_checkpoint(model, optimizer, scheduler, train_loss_evaluator, checkpoint_dir)

        if getattr(config, 'finetune_params', False):
            if start_epoch == 0:
                pretrained_checkpoint_dir = os.path.join(os.path.abspath(config.checkpoint_dir), config.finetune_params.pretrained_model_name)
                best_train_loss = float('inf')
                best_dev_loss = float('inf')
                best_test_loss = float('inf')
                start_epoch, wandb_run_id = self.load_latest_checkpoint(model, optimizer, scheduler, train_loss_evaluator, pretrained_checkpoint_dir, pretrained=True)
                # self.load_best_model(model, pretrained_checkpoint_dir, model_type='dev')
            
            set_trainable_layers(model, config.finetune_params.trainable_layers)

        # Initialize Weights and Biases logging
        if log_to_wandb:
            self.initialize_wandb(wandb_run_id, config_manager)

        # model = torch.compile(model, mode='max-autotune')
        # model = torch.compile(model, backend='eager', mode='default')
        # Main training loop
        for epoch in range(start_epoch+1, epochs+1):
            print(f'\n############### Epoch {epoch} Start ###############\n')
            logging.info(f"Epoch {epoch} / {epochs}")

            # Train
            self.train_epoch(epoch, model, train_dataloader, train_loss_evaluator, optimizer, scheduler, args, log_to_wandb, config)
            avg_train_loss = np.mean(train_loss_evaluator.losses_reported_metrics)
            best_train_loss = self.save_best_model(model, epoch, avg_train_loss, best_train_loss, checkpoint_dir, model_type='train')
            logging.info('Train Set Evaluation: ')
            train_loss_evaluator.print_report(args, reset=True)
            
            # Evaluate Dev
            self.evaluate_epoch(epoch, model, optimizer, dev_dataloader, dev_loss_evaluator, args, config)
            avg_dev_loss = np.mean(dev_loss_evaluator.losses_reported_metrics)
            best_dev_loss = self.save_best_model(model, epoch, avg_dev_loss, best_dev_loss, checkpoint_dir, model_type='dev')
            logging.info('Dev Set Evaluation: ')
            dev_loss_evaluator.print_report(args, reset=True, log_to_wandb=log_to_wandb, current_epoch=epoch)

            # Evaluate Test
            if test_flag:
                self.evaluate_epoch(epoch, model, optimizer, test_dataloader, test_loss_evaluator, args, config)
                avg_test_loss = np.mean(test_loss_evaluator.losses_reported_metrics)
                best_test_loss = self.save_best_model(model, epoch, avg_test_loss, best_test_loss, checkpoint_dir, model_type='test')
                logging.info('Test Set Evaluation: ')
                test_loss_evaluator.print_report(args, reset=True, log_to_wandb=log_to_wandb, current_epoch=epoch)

            # Save the checkpoint
            self.save_checkpoint(model, optimizer, scheduler, train_loss_evaluator, epoch, checkpoint_dir)

            print(f'\n############### Epoch {epoch} End ###############\n')
        return True

    def set_multiprocessing_start_method(self):
        """
        Set the multiprocessing start method to 'spawn' if not already set.
        """
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
            try:
                mp.set_start_method('spawn')
                logging.info("Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e:
                logging.warning(f"Could not set multiprocessing start method to 'spawn': {e}")
        else:
            logging.info(f"Multiprocessing start method is already set to '{current_method}'.")

    def get_dataloader_evaluator(self, dataset_path, split, config, geometry, device, data_loading_workers):
        """
        Create the DataLoader and LossEvaluator for a given dataset split.
        """
        loss_params = config.loss_params if hasattr(config, 'loss_params') else None
        data_params = config.data_params if hasattr(config, 'data_params') else ValueError("No data_params found in config.")
        
        # Initialize the dataset
        dataset = AddBiomechanicsDataset(
            dataset_path,
            data_params.history_len,
            device=device,
            stride=data_params.stride,
            geometry_folder=geometry,
            window_stride=data_params.window_stride,
            unbalanced_stride=getattr(data_params, 'unbalanced_stride', False) ,
            mode=split,
        )
        # Initialize the loss evaluator
        loss_evaluator = RegressionLossEvaluator(
            dataset=dataset, 
            split=split, 
            loss_params=loss_params if split == 'train' else None
        )
        # Create the DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=data_params.batch_size,
            shuffle=(split == 'train'),
            num_workers=data_loading_workers if split == 'train' else max(data_loading_workers, 1),
            # num_workers=data_loading_workers if split == 'train' else max(data_loading_workers//2, 1),
            persistent_workers=True
        )
        return dataloader, loss_evaluator
        
    def initialize_wandb(self, wandb_run_id, config_manager):
        """
        Initialize Weights and Biases logging.
        """
        logging.info('Initializing wandb...')
        if wandb_run_id:
            # Resume previous wandb run
            wandb.init(
                project="InferBiomechanics",
                config=config_manager.to_dict(),
                id=wandb_run_id,
                resume="allow"
            )
            logging.info(f"Resuming wandb run: {wandb_run_id}")
        else:
            # Start a new wandb run
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            new_run_id = f"{config_manager.config.model_name}_{timestamp}"
            wandb.init(
                project="InferBiomechanics",
                config=config_manager.to_dict(),
                id=new_run_id,
            )
        wandb.define_metric("train/step")
        wandb.define_metric("train/*", step_metric="train/step")
        wandb.define_metric("dev/step")
        wandb.define_metric("dev/*", step_metric="dev/step")
        print()

    def evaluate_epoch(self, epoch, model, optimizer, dataloader, loss_evaluator, args, config):
        """
        Evaluate the model.
        """
        print(f'\nEvaluating Set at Epoch {epoch}')
        model.eval()  # Set the model to evaluation mode
        if config.SchedulerFree:
            optimizer.eval()
        with torch.no_grad():
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", leave=True, dynamic_ncols=True)
            for i, batch in progress_bar:
                # Unpack the batch
                inputs, labels, batch_subject_indices, batch_trial_indices = batch
                # Forward pass with autocast if AMP is enabled
                if config.amp:
                    with autocast(config.device, dtype=torch.bfloat16):
                        outputs = model(inputs)
                        loss_evaluator(inputs, outputs, labels, batch_subject_indices, batch_trial_indices, args, compute_report=False)
                else:
                    outputs = model(inputs)
                    loss_evaluator(inputs, outputs, labels, batch_subject_indices, batch_trial_indices, args, compute_report=False)

                # Update the progress bar with metrics
                progress_bar.set_postfix({
                    'Force Err': f'{loss_evaluator.force_reported_metric:.4f} N/kg',
                    'CoP Err': f'{loss_evaluator.cop_reported_metric:.4f} m',
                    'Moment Err': f'{loss_evaluator.moment_reported_metric:.4f} Nm/kg'
                })

                if getattr(config.model_params, 'MoE_params', None):
                    clear_load_balancing_loss()
                    clear_router_zloss()

    def train_epoch(self, epoch, model, dataloader, loss_evaluator, optimizer, scheduler, args, log_to_wandb, config):
        """
        Train the model for one epoch.
        """
        print(f'\nRunning Train Epoch {epoch}')
        model.train()  # Set the model to training mode
        if config.SchedulerFree:
            optimizer.train()
        scaler = GradScaler(enabled=config.amp)

        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train Epoch {epoch}", leave=True, dynamic_ncols=True)
        for i, batch in progress_bar:
            # Unpack the batch
            inputs, labels, batch_subject_indices, batch_trial_indices = batch
            optimizer.zero_grad()   # Clear the gradients
            # Forward pass with autocast if AMP is enabled
            if config.amp:
                with autocast(config.device, dtype=torch.bfloat16):
                    outputs = model(inputs)
                    loss = loss_evaluator(inputs, outputs, labels, batch_subject_indices, batch_trial_indices, args, compute_report=(i % len(dataloader)//10 == 0), log_reports_to_wandb=log_to_wandb, logging_step=(epoch - 1 + (i / len(dataloader))))
            else:
                outputs = model(inputs)
                loss = loss_evaluator(inputs, outputs, labels, batch_subject_indices, batch_trial_indices, args, compute_report=(i % len(dataloader)//10 == 0), log_reports_to_wandb=log_to_wandb, logging_step=(epoch - 1 + (i / len(dataloader))))

            if getattr(config.model_params, 'MoE_params', None):
                lb_loss = batched_load_balancing_loss(model.moe_args)
                z_loss = batched_router_zloss(model.moe_args).mean()
                loss = loss + lb_loss + z_loss
                
                clear_load_balancing_loss()
                clear_router_zloss()

            # Backpropagation
            if config.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient clipping
            if getattr(config, 'max_grad_norm', None):
                if config.amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)

            # Update model parameters
            if config.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # Update the learning rate
            scheduler.step()

            # Update the progress bar with loss values
            if getattr(config.model_params, 'MoE_params', None):
                progress_bar.set_postfix({
                    'Force Loss': f'{loss_evaluator.loss_force:.4f}',
                    'CoP Loss': f'{loss_evaluator.loss_cop:.4f}',
                    'Moment Loss': f'{loss_evaluator.loss_moment:.4f}',
                    'LB Loss': f'{lb_loss:.4f}',
                    'Z Loss': f'{z_loss:.4f}'
                })
            else:
                progress_bar.set_postfix({
                    'Force Loss': f'{loss_evaluator.loss_force:.4f}',
                    'CoP Loss': f'{loss_evaluator.loss_cop:.4f}',
                    'Moment Loss': f'{loss_evaluator.loss_moment:.4f}'
                })
            

def get_optimizer(model: torch.nn.Module,
                  optimizer_params: Any) -> torch.optim.Optimizer:
    opt_type = optimizer_params.optimizer
    lr = optimizer_params.lr
    weight_decay = optimizer_params.weight_decay

    if opt_type == "Adam":
        print("[INFO] Using torch.optim.Adam")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=getattr(optimizer_params, 'betas', (0.9, 0.999)),
            weight_decay=weight_decay,
            eps=getattr(optimizer_params, 'eps', 1.0e-8)
        )

    elif opt_type == "AdamW":
        print("[INFO] Using torch.optim.AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=getattr(optimizer_params, 'betas', (0.9, 0.999)),
            weight_decay=weight_decay,
            eps=getattr(optimizer_params, 'eps', 1.0e-8)
        )

    elif opt_type == "ScheduleFreeAdamW":
        print("[INFO] Using Schedule-Free AdamW")
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=lr,
            betas=getattr(optimizer_params, 'betas', (0.9, 0.999)),
            eps=getattr(optimizer_params, 'eps', 1.0e-8),
            weight_decay=weight_decay,
            warmup_steps=getattr(optimizer_params, 'warmup_steps', 0),
            r=getattr(optimizer_params, 'r', 0.0),
            weight_lr_power=getattr(optimizer_params, 'weight_lr_power', 2.0)
        )

    elif opt_type == "ScheduleFreeRAdam":
        print("[INFO] Using Schedule-Free RAdam")
        optimizer = RAdamScheduleFree(
            model.parameters(),
            lr=lr,
            betas=getattr(optimizer_params, 'betas', (0.9, 0.999)),
            eps=getattr(optimizer_params, 'eps', 1.0e-8),
            weight_decay=weight_decay,
            r=getattr(optimizer_params, 'r', 0.0),
            weight_lr_power=getattr(optimizer_params, 'weight_lr_power', 2.0),
        )

    elif opt_type == "ScheduleFreeSGD":
        print("[INFO] Using Schedule-Free SGD")
        optimizer = SGDScheduleFree(
            model.parameters(),
            lr=lr,
            momentum=getattr(optimizer_params, 'momentum', 0.9),
            weight_decay=weight_decay,
            warmup_steps=getattr(optimizer_params, 'warmup_steps', 0),
            r=getattr(optimizer_params, 'r', 0.0),
            weight_lr_power=getattr(optimizer_params, 'weight_lr_power', 2.0)
        )

    else:
        raise ValueError(f"[ERROR] Unknown optimizer type: {opt_type}")

    return optimizer

def get_hf_scheduler(
    optimizer,
    scheduler_params,
    num_training_steps: int,
):

    if not scheduler_params:
        logging.warning("No scheduler_params found. Using 'constant' scheduler by default.")
        raw_type = "constant"
    else:
        raw_type = getattr(scheduler_params, "type", "").lower()
        if not raw_type:
            logging.warning("No scheduler type set. Using 'constant' scheduler by default.")
            raw_type = "constant"
    try:
        scheduler_type = SchedulerType(raw_type)
        if scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            logging.warning("`reduce_lr_on_plateau` scheduler is not supported. Using 'constant' instead.")
            scheduler_type = SchedulerType.CONSTANT
    except ValueError:
        logging.warning(f"Unsupported scheduler type: {raw_type}. Using 'constant' instead.")
        scheduler_type = SchedulerType.CONSTANT

    num_warmup_steps = int(num_training_steps * getattr(scheduler_params, "num_warmup_step_ratio", 5)/100)

    # 3) 스케줄러별 추가 파라미터 세팅
    scheduler_specific_kwargs = {}
    if scheduler_type == SchedulerType.COSINE_WITH_RESTARTS:
        # => get_cosine_with_hard_restarts_schedule_with_warmup
        scheduler_specific_kwargs["num_cycles"] = getattr(scheduler_params, "num_cycles", 1)

    elif scheduler_type == SchedulerType.COSINE_WITH_MIN_LR:
        # => get_cosine_with_min_lr_schedule_with_warmup
        min_lr = getattr(scheduler_params, "min_lr", None)
        min_lr_rate = getattr(scheduler_params, "min_lr_rate", None)
        if min_lr is not None:
            scheduler_specific_kwargs["min_lr"] = min_lr
        elif min_lr_rate is not None:
            scheduler_specific_kwargs["min_lr_rate"] = min_lr_rate
        scheduler_specific_kwargs["num_cycles"] = getattr(scheduler_params, "num_cycles", 0.5)

    elif scheduler_type == SchedulerType.WARMUP_STABLE_DECAY:
        # => get_wsd_schedule
        scheduler_specific_kwargs["num_stable_steps"] = int(num_training_steps * getattr(scheduler_params, "num_stable_step_ratio", 70)/100)
        scheduler_specific_kwargs["num_decay_steps"] = int(num_training_steps * getattr(scheduler_params, "num_decay_step_ratio", 25)/100)
        scheduler_specific_kwargs["min_lr_ratio"] = getattr(scheduler_params, "min_lr_ratio", 0)
        scheduler_specific_kwargs["num_cycles"] = getattr(scheduler_params, "num_cycles", 0.5)

    elif scheduler_type == SchedulerType.POLYNOMIAL:
        # => get_polynomial_decay_schedule_with_warmup
        lr_end = getattr(scheduler_params, "lr_end", None)
        power = getattr(scheduler_params, "power", None)
        if lr_end is not None:
            scheduler_specific_kwargs["lr_end"] = lr_end
        if power is not None:
            scheduler_specific_kwargs["power"] = power

    elif scheduler_type == SchedulerType.INVERSE_SQRT:
        timescale = getattr(scheduler_params, "timescale", None)
        if timescale is not None:
            scheduler_specific_kwargs["timescale"] = timescale

    scheduler = get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=scheduler_specific_kwargs
    )

    logging.info(
        f"Using HuggingFace Scheduler: {scheduler_type.value} "
        f"(warmup={num_warmup_steps}, total={num_training_steps}, extra={scheduler_specific_kwargs})"
    )
    return scheduler

def set_trainable_layers(model, trainable_patterns=None):
    """
    Sets only specified layers as trainable while freezing all others.
    
    Args:
        model: Target model
        trainable_patterns: List of layer patterns to make trainable
            - None: All layers trainable (default)
            - 'none': No layers trainable (freeze all)
            - ['head', '*ffn']: Layers containing 'head' and any layers containing 'ffn'
    """
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
        
    # Return early if no layers should be trainable
    if trainable_patterns == 'none':
        logging.info("All layers frozen (trainable_patterns='none')")
        return
    
    # If None, make all parameters trainable
    if trainable_patterns is None or trainable_patterns == ['all']:
        for param in model.parameters():
            param.requires_grad = True
        logging.info("All layers are trainable")
        return

    # Set specified layers as trainable
    trainable_count = 0
    total_count = 0
    trainable_params = 0
    total_params = 0
    trainable_layers = []

    for name, param in model.named_parameters():
        total_count += 1
        total_params += param.numel()
        for pattern in trainable_patterns:
            if fnmatch.fnmatch(name, pattern):
                param.requires_grad = True
                trainable_count += 1
                trainable_params += param.numel()
                trainable_layers.append(name)
                break
    
    # Print fine-tuning status
    print("\n" + "="*50)
    print(f"Fine-tuning Status:")
    print(f"- Total layers: {total_count}")
    print(f"- Trainable layers: {trainable_count} ({trainable_count/total_count*100:.1f}%)")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"- Frozen parameters: {total_params-trainable_params:,} ({(1-trainable_params/total_params)*100:.1f}%)")
    print("\nTrainable layers:")
    for layer in trainable_layers:
        print(f"  - {layer}")
    print("="*50 + "\n")
    
    logging.info(f"Set {trainable_count}/{total_count} layers as trainable ({trainable_count/total_count*100:.1f}%)")
    logging.info(f"Trainable parameters: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.1f}%)")
