import argparse
import os
import torch
from models.AnalyticalBaseline import AnalyticalBaseline
from data.AddBiomechanicsDataset import AddBiomechanicsDataset
from typing import List
import logging
import wandb

class AbstractCommand:
    """
    All of our different activities inherit from this class. This class defines the interface for a CLI command, so
    that it's convenient to split commands across files. It also carries shared logic for loading / saving models, etc.
    """
    def register_subcommand(self, subparsers: argparse._SubParsersAction):
        pass

    def run(self, args: argparse.Namespace) -> bool:
        pass

    def ensure_geometry(self, geometry: str):
        if geometry is None:
            # Check if the "./Geometry" folder exists, and if not, download it
            if not os.path.exists('./Geometry'):
                print('Downloading the Geometry folder from https://addbiomechanics.org/resources/Geometry.zip')
                exit_code = os.system('wget https://addbiomechanics.org/resources/Geometry.zip')
                if exit_code != 0:
                    print('ERROR: Failed to download Geometry.zip. You may need to install wget. If you are on a Mac, '
                          'try running "brew install wget"')
                    return False
                os.system('unzip ./Geometry.zip')
                os.system('rm ./Geometry.zip')
            geometry = './Geometry'
        print('Using Geometry folder: ' + geometry)
        geometry = os.path.abspath(geometry)
        if not geometry.endswith('/'):
            geometry += '/'
        return geometry
    
    def load_latest_checkpoint(self, model, optimizer, scheduler, evaluator, checkpoint_dir="../checkpoints", pretrained=False):
        if not os.path.exists(checkpoint_dir):
            print("Checkpoint directory does not exist!")
            return 0, None
        
        # Get all the checkpoint files that start with 'epoch_'
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch_") and f.endswith(".pt")]

        # If there are no checkpoints, return
        if not checkpoints:
            print("No checkpoints available!")
            return 0, None

        # Sort the files based on the epoch in their filenames
        checkpoints.sort(key=lambda x: (int(x.split('_')[1].split('.')[0])))

        # Get the path of the latest checkpoint
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

        # Load the checkpoint
        checkpoint = torch.load(latest_checkpoint, weights_only=False)

        # Load the model, optimizer, scheduler, and evaluator states
        model.load_state_dict(checkpoint['model_state_dict'])
        if pretrained:
            print(f"Loaded checkpoint from epoch Pretrained Model")
            return 0, None
        else:
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if evaluator is not None and 'evaluator_state_dict' in checkpoint:
                evaluator.load_state_dict(checkpoint['evaluator_state_dict'])

            epoch = checkpoint['epoch']
            wandb_run_id = checkpoint.get('wandb_run_id', None)

            print(f"Loaded checkpoint from epoch {epoch}")
            return epoch, wandb_run_id

    def load_best_model(self, model, checkpoint_dir="../checkpoints", opt='model', model_type='train'):
        """
        Method to load the best model.
        If best_model.pt exists, load the checkpoint, otherwise return None.
        """
        best_model_path = os.path.join(checkpoint_dir, f'best_model_{model_type}.pt')

        if not os.path.exists(best_model_path):
            print(f"{best_model_path} checkpoint does not exist!")
            return float('inf')

        # Load the best model checkpoint
        checkpoint = torch.load(best_model_path, weights_only=False)

        if opt == 'loss':
            best_loss = checkpoint['loss']
            return best_loss
        else:
        # Load the model, optimizer, and scheduler states
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']

            print(f"Loaded best model from epoch {epoch} with loss {best_loss:.4f}")
            return
    
    def save_checkpoint(self, model, optimizer, scheduler, evaluator, epoch, checkpoint_dir):
        """
        Saves the current model state and evaluator to a checkpoint and deletes the previous checkpoint.

        :param model: The model to save.
        :param optimizer: The optimizer state.
        :param scheduler: The scheduler state.
        :param evaluator: The RegressionLossEvaluator instance.
        :param epoch: The current epoch.
        :param checkpoint_dir: The directory to save the checkpoint.
        """
        model_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pt')
        
        # If the directory does not exist, create it
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'evaluator_state_dict': evaluator.state_dict(),  # Save RegressionLossEvaluator state
            'wandb_run_id': wandb.run.id if wandb.run else None
        }

        if scheduler is not None:
            checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()

        # Save the current checkpoint
        torch.save(checkpoint_dict, model_path)
        
        # If there is a previous checkpoint file, remove it (excluding the current file)
        for filename in os.listdir(checkpoint_dir):
            if filename != f'epoch_{epoch}.pt' and filename.startswith('epoch_'):
                os.remove(os.path.join(checkpoint_dir, filename))

    def save_best_model(self, model, epoch, avg_loss, best_loss, checkpoint_dir, model_type='train'):
        """
        Saves the best model.

        :param model: The model object.
        :param epoch: The current epoch.
        :param avg_loss: The average loss of the current epoch.
        :param best_loss: The best loss so far.
        :param checkpoint_dir: The checkpoint directory.
        :return: The updated best loss value.
        """
        best_model_path = os.path.join(checkpoint_dir, f'best_model_{model_type}.pt')

        # If the directory does not exist, create it
        os.makedirs(checkpoint_dir, exist_ok=True)

        if avg_loss < best_loss:
            torch.save({
                'loss': avg_loss,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
            }, best_model_path)
            print(f'New best {model_type} model saved at epoch {epoch} with loss {avg_loss:.4f}')
            return avg_loss

        return best_loss
    
    def print_model_summary(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params}, Trainable Parameters: {trainable_params}")

    @staticmethod
    def sliding_window_inference(model, inputs, window_size, amp, device, overlap_ratio=0.9, momentum=0.9):
        """
        Performs sliding window inference with momentum applied to overlapping regions,
        ensuring all frames up to the end of the sequence are processed.
        
        Args:
            model: The regression model
            inputs: Dictionary of input tensors
            window_size: Size of the sliding window
            amp: Whether to use automatic mixed precision
            device: Device to run inference on
            overlap_ratio: Ratio of window to overlap (between 0.0 and 1.0)
                        0.0 means no overlap, 0.5 means 50% overlap
            momentum: Momentum coefficient for combining predictions (between 0.0 and 1.0)
        
        Returns:
            outputs_combined: dict of outputs with shape [1, total_seq_length, ...]
        """
        momentum = min(1.0, max(0.0, momentum))
        overlap_ratio = min(1.0, max(0.0, overlap_ratio))

        seq_length = list(inputs.values())[0].shape[1]
        outputs_combined = {}
        
        # Calculate stride based on overlap_ratio
        stride = max(1, int(window_size * (1 - overlap_ratio)))
        
        # Process first window (0 to window_size-1 frames)
        with torch.no_grad():
            if amp:
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    out = model({k: v[:, :window_size] for k, v in inputs.items()})
            else:
                out = model({k: v[:, :window_size] for k, v in inputs.items()})
        
        # Initialize outputs with first window predictions
        for key, tensor in out.items():
            outputs_combined[key] = tensor.clone()
        
        # Generate a list of all window start positions to ensure we cover the entire sequence
        window_starts = list(range(stride, seq_length - window_size + 1, stride))
        
        # Add a final window if needed to ensure we reach the end of the sequence
        if window_starts and window_starts[-1] + window_size < seq_length:
            final_start = seq_length - window_size
            if final_start > window_starts[-1]:
                window_starts.append(final_start)
        
        # Process all windows
        for i in window_starts:
            window_inputs = {k: v[:, i:i+window_size] for k, v in inputs.items()}
            
            with torch.no_grad():
                if amp:
                    with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                        out_window = model(window_inputs)
                else:
                    out_window = model(window_inputs)
            
            for key, tensor in out_window.items():
                current_output_len = outputs_combined[key].shape[1]
                
                # Determine overlap region and new region
                overlap_start_idx = i
                overlap_end_idx = min(current_output_len, i + window_size)
                overlap_length = overlap_end_idx - overlap_start_idx
                
                # Apply momentum to overlapping region if it exists
                if overlap_length > 0:
                    # Get corresponding tensors for overlap regions
                    prev_tensor = outputs_combined[key][:, overlap_start_idx:overlap_end_idx]
                    new_tensor = tensor[:, :overlap_length]
                    
                    # Apply momentum
                    combined_tensor = (1 - momentum) * prev_tensor + momentum * new_tensor
                    
                    # Update the overlapping region
                    outputs_combined[key][:, overlap_start_idx:overlap_end_idx] = combined_tensor
                
                # Append non-overlapping region
                if overlap_end_idx < i + window_size:
                    non_overlap_tensor = tensor[:, overlap_length:].clone()
                    outputs_combined[key] = torch.cat([outputs_combined[key], non_overlap_tensor], dim=1)
        
        return outputs_combined
