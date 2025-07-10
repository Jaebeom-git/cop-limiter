import torch
from data.AddBiomechanicsDataset import AddBiomechanicsDataset, OutputDataKeys, InputDataKeys
from typing import Dict, List, Optional
import numpy as np
import wandb
import logging
import matplotlib.pyplot as plt
import os
import argparse
from loss.Losses import calculate_diff, se_loss, mean_norm_error, GHMR, DenseLoss, TCLoss, MPJVE
from loss.Weight_Calculator import ForceWeightCalculator, ForceMaskCalculator, CoPWeightCalculator

components = {
    0: "left-x",
    1: "left-y",
    2: "left-z",
    3: "right-x",
    4: "right-y",
    5: "right-z"
}

class RegressionLossEvaluator:
    dataset: AddBiomechanicsDataset

    losses: List[torch.Tensor]
    force_losses: List[torch.Tensor]
    moment_losses: List[torch.Tensor]
    wrench_losses: List[torch.Tensor]
    cop_losses: List[torch.Tensor]

    force_reported_metrics: List[float]
    moment_reported_metrics: List[float]
    cop_reported_metrics: List[float]
    # wrench_reported_metrics: List[float]
    tau_reported_metrics: List[float]
    tau_reported_metrics_std: List[float]   # joint torque std
    # com_acc_reported_metrics: List[float]

    def __init__(self, dataset: AddBiomechanicsDataset, split: str, loss_params=None):
        self.dataset = dataset
        self.split = split

        # Loss parameters
        if loss_params is not None:
            self.scale_params = getattr(loss_params, 'scale_params', None)
            self.force_weight_params = getattr(loss_params, 'force_weight_params', None)
            self.cop_weight_params = getattr(loss_params, 'cop_weight_params', None)
            self.GHMR_params = getattr(loss_params, 'GHMR_params', None)
            self.DenseLoss_params = getattr(loss_params, 'DenseLoss_params', None)
            self.OHEM_params = getattr(loss_params, 'OHEM_params', None)
            self.force_mask_threshold = getattr(loss_params, 'force_mask_threshold', 1.0)
        else:
            self.scale_params = None
            self.force_weight_params = None
            self.cop_weight_params = None
            self.GHMR_params = None
            self.DenseLoss_params = None
            self.OHEM_params = None
            self.force_mask_threshold = 1.0

        # Force mask calculator
        self.force_mask_calculator = ForceMaskCalculator(threshold=self.force_mask_threshold)

        # Weight scales
        if self.scale_params is not None:
            self.force_weight_scale = getattr(self.scale_params, 'force', 1.0)
            self.moment_weight_scale = getattr(self.scale_params, 'moment', 1.0)
            self.cop_weight_scale = getattr(self.scale_params, 'cop', 1.0)
            self.TCLoss_weight_scale = getattr(self.scale_params, 'TCLoss', 0.0)
            self.MPJVE_weight_scale = getattr(self.scale_params, 'MPJVE', 0.0)
        else:
            self.force_weight_scale = 1.0
            self.moment_weight_scale = 1.0
            self.cop_weight_scale = 1.0
            self.TCLoss_weight_scale = 0.0
            self.MPJVE_weight_scale = 0.0

        # Weight calculators
        self.force_weight_calculator = ForceWeightCalculator(**vars(self.force_weight_params)) if self.force_weight_params else None
        self.cop_weight_calculator = CoPWeightCalculator(**vars(self.cop_weight_params)) if self.cop_weight_params else None

        # OHEM
        if self.OHEM_params is not None:
            self.use_ohem = True
            self.use_ohem_fraction = self.OHEM_params.ohem_fraction
        else:
            self.use_ohem = False

        # GHMR
        self.ghmr_forces = [GHMR(mu=self.GHMR_params.mu, bins=self.GHMR_params.bins, momentum=self.GHMR_params.momentum, momentum_increment=self.GHMR_params.momentum_increment, loss_weight=self.force_weight_scale) for _ in range(6)] if self.GHMR_params else None
        self.ghmr_moments = [GHMR(mu=self.GHMR_params.mu, bins=self.GHMR_params.bins, momentum=self.GHMR_params.momentum, momentum_increment=self.GHMR_params.momentum_increment, loss_weight=self.moment_weight_scale) for _ in range(6)] if self.GHMR_params else None
        self.ghmr_cops = [GHMR(mu=self.GHMR_params.mu, bins=self.GHMR_params.bins, momentum=self.GHMR_params.momentum, momentum_increment=self.GHMR_params.momentum_increment, loss_weight=self.cop_weight_scale) for _ in range(6)] if self.GHMR_params else None

        # DenseWeight
        self.denseloss_forces = DenseLoss(**vars(self.DenseLoss_params), loss_weight=self.force_weight_scale) if self.DenseLoss_params else None
        self.denseloss_moments = DenseLoss(**vars(self.DenseLoss_params), loss_weight=self.moment_weight_scale) if self.DenseLoss_params else None
        self.denseloss_cops = DenseLoss(**vars(self.DenseLoss_params), loss_weight=self.cop_weight_scale) if self.DenseLoss_params else None

        # Aggregating losses across batches for dev set evaluation
        self.losses = []
        self.force_losses = []
        self.moment_losses = []
        self.cop_losses = []

        # Aggregating reported metrics for dev set evaluation
        self.force_reported_metrics = []
        self.moment_reported_metrics = []
        self.cop_reported_metrics = []
        self.tau_reported_metrics = []
        self.tau_reported_metrics_std = []
        self.losses_reported_metrics = []

    def __call__(self,
                 inputs: Dict[str, torch.Tensor],
                 outputs: Dict[str, torch.Tensor],
                 labels: Dict[str, torch.Tensor],
                 batch_subject_indices: List[int],
                 batch_trial_indices: List[int],
                 args: argparse.Namespace,
                 compute_report: bool = False,
                 log_reports_to_wandb: bool = False,
                 logging_step: Optional[float] = None,
                 analyze: bool = False,
                 plot_path_root: str = 'outputs/plots') -> torch.Tensor:

        device = next(iter(outputs.values())).device
        num_batches, sequence_length, _ = outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME].shape
        ############################################################################
        # Step 0: Compute the force weight
        ############################################################################

        # Compute force_weight dynamically
        if self.force_weight_calculator is not None:
            force_weight_tensor = self.force_weight_calculator.compute_weight(labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].to(device))  # [B, T, 6]
        else:
            force_weight_tensor = None

        ############################################################################
        # Step 1: Compute the loss
        ############################################################################

        # 1.1. Compute the force loss
        predicted_forces = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME]
        target_forces = labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].to(device)
        force_diff = calculate_diff(predicted_forces, target_forces)
        force_se = se_loss(force_diff, reduction='none')
        self.force_losses.append(torch.mean(force_se, dim=(0, 1)).detach())

        if self.use_ohem:
            with torch.no_grad():
                weighted_force_se = force_se * force_weight_tensor if force_weight_tensor is not None else force_se
                force_loss_per_sample = torch.mean(weighted_force_se, dim=1) * self.force_weight_scale
                per_sample_total_loss = torch.sum(force_loss_per_sample[:, args.predict_grf_components], dim=1)

                k = max(int(num_batches * self.use_ohem_fraction), 1)
                _, indices = torch.topk(per_sample_total_loss, k, largest=True)
            force_se = force_se[indices]
            force_weight_tensor = force_weight_tensor[indices] if force_weight_tensor is not None else None
            target_forces = target_forces[indices]

        if self.denseloss_forces is not None:
            force_component_losses = [
                self.denseloss_forces(force_se[:, :, i], target_forces[:, :, i])
                for i in range(6)
            ]
            force_loss = torch.stack(force_component_losses)
        elif self.ghmr_forces is not None:
            force_component_losses = [
                self.ghmr_forces[i](
                    force_se[:, :, i], force_diff[:, :, i], 
                    weight=force_weight_tensor[:, :, i] if force_weight_tensor is not None else None
                )
                for i in range(6)
            ]
            force_loss = torch.stack(force_component_losses)
        else:
            weighted_force_se = force_se * force_weight_tensor if force_weight_tensor is not None else force_se
            force_loss = torch.mean(weighted_force_se, dim=(0, 1)) * self.force_weight_scale

        # 1.2. Compute the moment loss
        predicted_moments = outputs[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME]
        target_moments = labels[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME].to(device)
        if self.use_ohem:
            predicted_moments = predicted_moments[indices]
            target_moments = target_moments[indices]
        moment_diff = calculate_diff(predicted_moments, target_moments)
        moment_se = se_loss(moment_diff, reduction='none')
        self.moment_losses.append(torch.mean(moment_se, dim=(0, 1)).detach())
        if self.denseloss_moments is not None:
            moment_component_losses = [
                self.denseloss_moments(moment_se[:, :, i], target_moments[:, :, i])
                for i in range(6)
            ]
            moment_loss = torch.stack(moment_component_losses)
        elif self.ghmr_moments:
            moment_component_losses = [
                self.ghmr_moments[i](
                    moment_se[:, :, i], moment_diff[:, :, i], 
                    weight=force_weight_tensor[:, :, i] if force_weight_tensor is not None else None
                )
                for i in range(6)
            ]
            moment_loss = torch.stack(moment_component_losses)
        else:
            weighted_moment_se = moment_se * force_weight_tensor if force_weight_tensor is not None else moment_se
            moment_loss = torch.mean(weighted_moment_se, dim=(0, 1)) * self.moment_weight_scale

        # 1.3. Compute the CoP loss
        with torch.no_grad():
            cop_mask_tensor = self.force_mask_calculator.get_mask_by_threes(target_forces)
        
        predicted_cop = outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME]
        true_cop = labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME].to(device)
        if self.use_ohem:
            predicted_cop = predicted_cop[indices]
            true_cop = true_cop[indices]
        predicted_cop = predicted_cop * cop_mask_tensor
        true_cop = true_cop * cop_mask_tensor
        
        if self.cop_weight_calculator is not None:
            cop_weight_tensor = self.cop_weight_calculator.compute_weight(predicted_cop, true_cop)  # [B, T, 6]
        else:
            cop_weight_tensor = None

        cop_diff = calculate_diff(predicted_cop, true_cop)
        cop_se = se_loss(cop_diff, reduction='none')
        self.cop_losses.append(torch.mean(cop_se, dim=(0, 1)).detach())
        if self.denseloss_cops is not None:
            cop_component_losses = [
                self.denseloss_cops(cop_se[:, :, i], true_cop[:, :, i])
                for i in range(6)
            ]
            cop_loss = torch.stack(cop_component_losses)
        elif self.ghmr_cops is not None:
            cop_component_losses = [
                self.ghmr_cops[i](
                    cop_se[:, :, i], cop_diff[:, :, i], 
                    weight=cop_weight_tensor[:, :, i] if cop_weight_tensor is not None else None
                )
                for i in range(6)
            ]
            cop_loss = torch.stack(cop_component_losses)
        else:
            weighted_cop_se = cop_se * cop_weight_tensor if cop_weight_tensor is not None else cop_se
            cop_loss = torch.mean(weighted_cop_se, dim=(0, 1)) * self.cop_weight_scale

        # 1.4. Sum the losses based on user-specified components
        # MSE losses are vectors; sum over specified components
        loss_force = torch.sum(force_loss[args.predict_grf_components])
        loss_moment = torch.sum(moment_loss[args.predict_moment_components])
        loss_cop = torch.sum(cop_loss[args.predict_cop_components])
        loss = loss_force + loss_moment + loss_cop

        # 1.5. Apply Temporal Consistency Loss if enabled
        if self.TCLoss_weight_scale > 0:
            if self.use_ohem:
                forces_tc = predicted_forces[indices]
                moments_tc = predicted_moments[indices]
                cop_tc = predicted_cop[indices]
            else:
                forces_tc = predicted_forces
                moments_tc = predicted_moments
                cop_tc = predicted_cop

            tc_loss_forces = TCLoss(forces_tc) * self.force_weight_scale
            tc_loss_moments = TCLoss(moments_tc) * self.moment_weight_scale
            tc_loss_cop = TCLoss(cop_tc) * self.cop_weight_scale
            tc_loss_total = tc_loss_forces + tc_loss_moments + tc_loss_cop
            loss = loss + self.TCLoss_weight_scale * tc_loss_total

        # 1.6. Apply MPJVE Loss if enabled
        if self.MPJVE_weight_scale > 0:
            if self.use_ohem:
                forces_pred_temp = predicted_forces[indices]
                forces_target_temp = target_forces[indices]
                moments_pred_temp = predicted_moments[indices]
                moments_target_temp = target_moments[indices]
                cop_pred_temp = predicted_cop[indices]
                cop_target_temp = true_cop[indices]
            else:
                forces_pred_temp = predicted_forces
                forces_target_temp = target_forces
                moments_pred_temp = predicted_moments
                moments_target_temp = target_moments
                cop_pred_temp = predicted_cop
                cop_target_temp = true_cop

            mpjve_loss_forces = MPJVE(forces_pred_temp, forces_target_temp) * self.force_weight_scale
            mpjve_loss_moments = MPJVE(moments_pred_temp, moments_target_temp) * self.moment_weight_scale
            mpjve_loss_cop = MPJVE(cop_pred_temp, cop_target_temp) * self.cop_weight_scale
            mpjve_loss_total = mpjve_loss_forces + mpjve_loss_moments + mpjve_loss_cop
            loss = loss + self.MPJVE_weight_scale * mpjve_loss_total

        self.losses.append(loss.item())
        self.loss_force = loss_force.item()
        self.loss_moment = loss_moment.item()
        self.loss_cop = loss_cop.item()

        ############################################################################
        # Step 2: Compute report data, if we are asked to do so
        ############################################################################

        # 2.1. Initialize paper-reported values we will send to wandb, if requested
        tau_reported_metric: Optional[float] = None

        with torch.no_grad():
            # 2.2. Compute the norm errors for the force, moment, CoP, and wrench vectors
            self.force_reported_metric: float = mean_norm_error(force_diff).item()
            self.moment_reported_metric: float = mean_norm_error(moment_diff).item()
            self.cop_reported_metric: float = mean_norm_error(cop_diff).item()
            
            loss_reported_metric = self.force_reported_metric + self.moment_reported_metric + self.cop_reported_metric

            if compute_report:
                # 2.3. Manually compute the inverse dynamics torque errors frame-by-frame
                tau_reported_metric = 0.0
                tau_reported_metric_std = 0.0
                batch_indices = range(num_batches)
                skels = [self.dataset.skeletons[batch_subject_indices[batch]] for batch in batch_indices]
                skel_masses = [skel.getMass() for skel in skels]
                skel_masses = np.array(skel_masses)

                positions = inputs[InputDataKeys.POS].cpu().numpy()   # shape: [B, T, D]
                velocities = inputs[InputDataKeys.VEL].cpu().numpy()    # shape: [B, T, D]
                accs = inputs[InputDataKeys.ACC].cpu().numpy()          # shape: [B, T, D]
                pred_forces = outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].float().cpu().numpy()   # shape: [B, T, 6]
                pred_moments = outputs[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME].float().cpu().numpy()   # shape: [B, T, 6]
                pred_cops = outputs[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME].float().cpu().numpy()   # shape: [B, T, 6]
                labels_forces = labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].cpu().numpy()   # shape: [B, T, 6]
                labels_moments = labels[OutputDataKeys.GROUND_CONTACT_TORQUES_IN_ROOT_FRAME].cpu().numpy()   # shape: [B, T, 6]
                labels_cops = labels[OutputDataKeys.GROUND_CONTACT_COPS_IN_ROOT_FRAME].cpu().numpy()   # shape: [B, T, 6]
                # labels_tau = labels[OutputDataKeys.TAU].cpu().numpy()   # shape: [B, T, D], (optimized residual force)

                # tau_metric_joints = ['hip', 'knee', 'ankle', 'subtalar']
                for batch, skel, skel_mass in zip(batch_indices, skels, skel_masses):
                    # dof_names = []
                    # for i in range(skel.getNumDofs()):
                    #     dof_names.append(skel.getDofByIndex(i).getName())
                    # selected_joint_idx = [i for i, name in enumerate(dof_names) if any(keyword in name for keyword in tau_metric_joints)]

                    batch_tau_reported_metric = 0.0
                    batch_tau_reported_metric_std = 0.0
                    T = positions.shape[1]
                    for t in range(T):
                        skel.setPositions(positions[batch, t, :])
                        skel.setVelocities(velocities[batch, t, :])
                        acc = accs[batch, t, :]
                        contact_bodies = self.dataset.skeletons_contact_bodies[batch_subject_indices[batch]]
                        
                        # Compute the pred
                        batch_forces = pred_forces[batch, t, :] * skel_mass
                        batch_moments = pred_moments[batch, t, :] * skel_mass
                        batch_cops = pred_cops[batch, t, :]
                        batch_rxF = np.cross(batch_cops.reshape(2, 3), batch_forces.reshape(2, 3))
                        
                        # Wrench Moments: GRM (Free Moments) + rxF (CoP x GRF)
                        contact_wrench_pred_list = [
                            np.concatenate([batch_moments[:3] + batch_rxF[0], batch_forces[:3]]),
                            np.concatenate([batch_moments[3:] + batch_rxF[1], batch_forces[3:]])
                        ]
                        pred_tau = skel.getInverseDynamicsFromPredictions(acc, contact_bodies, contact_wrench_pred_list, np.zeros(6))

                        # # Error: labels_tau(optimized residual force) - pred_tau(set zero for residual force)
                        # tau_error = pred_tau - labels_tau[batch, t, :]

                        # Compute the true
                        batch_forces = labels_forces[batch, t, :] * skel_mass
                        batch_moments = labels_moments[batch, t, :] * skel_mass
                        batch_cops = labels_cops[batch, t, :]
                        batch_rxF = np.cross(batch_cops.reshape(2, 3), batch_forces.reshape(2, 3))

                        # Wrench Moments: GRM (Free Moments) + rxF (CoP x GRF)
                        contact_wrench_true_list = [
                            np.concatenate([batch_moments[:3] + batch_rxF[0], batch_forces[:3]]),
                            np.concatenate([batch_moments[3:] + batch_rxF[1], batch_forces[3:]])
                        ]
                        true_tau = skel.getInverseDynamicsFromPredictions(acc, contact_bodies, contact_wrench_true_list, np.zeros(6))

                        tau_error = pred_tau - true_tau

                        # Exclude root residual from error
                        # normalized_errors = np.abs(tau_error[selected_joint_idx]) / skel_mass
                        normalized_errors = np.abs(tau_error[6:]) / skel_mass
                        batch_tau_reported_metric += np.mean(normalized_errors)
                        batch_tau_reported_metric_std += np.std(normalized_errors)

                    batch_tau_reported_metric /= T
                    batch_tau_reported_metric_std /= T
                    tau_reported_metric += batch_tau_reported_metric
                    tau_reported_metric_std += batch_tau_reported_metric_std
                
                tau_reported_metric /= num_batches
                tau_reported_metric_std /= num_batches
                
                self.tau_reported_metrics.append(tau_reported_metric)
                self.tau_reported_metrics_std.append(tau_reported_metric_std) 

            # 2.4. Keep track of the reported metrics for reporting averages across the entire dev set
            self.force_reported_metrics.append(self.force_reported_metric)
            self.moment_reported_metrics.append(self.moment_reported_metric)
            self.cop_reported_metrics.append(self.cop_reported_metric)
            self.losses_reported_metrics.append(loss_reported_metric)

        ############################################################################
        # Step 3: Log reports to wandb and plot results, if requested
        ############################################################################

        # 3.1. If requested, log the reports to Weights and Biases
        if log_reports_to_wandb:
            self.log_to_wandb(args, self.force_losses[-1], self.cop_losses[-1], self.moment_losses[-1], loss, self.force_reported_metric, self.cop_reported_metric, self.moment_reported_metric,tau_reported_metric, loss_reported_metric, step=logging_step)

        # 3.2. If requested, plot the results
        if analyze:
            self.plot_ferror = ((outputs[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME] - labels[OutputDataKeys.GROUND_CONTACT_FORCES_IN_ROOT_FRAME].to(device)) ** 2)[:, -1, :].reshape(-1, 6).detach().numpy()
            for i in args.predict_grf_components:
                plt.clf()
                plt.plot(self.plot_ferror[:, i])
                plt.savefig(os.path.join(plot_path_root,
                                         f"{os.path.basename(self.dataset.subject_paths[batch_subject_indices[0]])}_{self.dataset.subjects[batch_subject_indices[0]].getTrialName(batch_trial_indices[0])}_grferror{components[i]}.png"))
                
        return loss

    def log_to_wandb(self,
                     args: argparse.Namespace,
                     force_loss: torch.Tensor,
                     cop_loss: torch.Tensor,
                     moment_loss: torch.Tensor,
                     loss: torch.Tensor,
                     # IMPORTANT: THESE ARE NOT THE SAME AS THE LOSS VALUES ABOVE! These compute errors per vector pair
                     # as a simple norm, and take the mean of that. The above losses are squared errors, and are summed.
                     # If we were to then square-root the above losses, we would get higher values than the ones
                     # reported here:
                     force_reported_metric: Optional[float],
                     cop_reported_metric: Optional[float],
                     moment_reported_metric: Optional[float],
                     tau_reported_metric: Optional[float],
                     loss_reported_metric: Optional[float],
                     step: Optional[float] = None):  # Added step parameter
        
        # MSE losses are vectors
        report: Dict[str, float] = {
            **{f'{self.split}/force_rmse/{components[i]}': force_loss[i].item() ** 0.5 for i in
               args.predict_grf_components},
            **{f'{self.split}/cop_rmse/{components[i]}': cop_loss[i].item() ** 0.5 for i in
               args.predict_cop_components},
            **{f'{self.split}/moment_rmse/{components[i]}': moment_loss[i].item() ** 0.5 for i in
               args.predict_moment_components},
            f'{self.split}/loss': loss.item()
        }
            
        if force_reported_metric is not None:
            report[f'{self.split}/reports/Force Err (N per kg)'] = force_reported_metric
        if cop_reported_metric is not None:
            report[f'{self.split}/reports/CoP Err (m)'] = cop_reported_metric
        if moment_reported_metric is not None:
            report[f'{self.split}/reports/Moment Err (Nm per kg)'] = moment_reported_metric
        if tau_reported_metric is not None:
            report[f'{self.split}/reports/Non-root Joint Torques (Inverse Dynamics) Err (Nm per kg)'] = tau_reported_metric
        
        # Norm-based loss_reported_metric (mean of force, cop, moment errors)
        if loss_reported_metric is not None:
            report[f'{self.split}/reports/Loss (Norm Err)'] = loss_reported_metric

        if step is not None:
            report[f'{self.split}/step'] = step

        # Log to wandb with the specified step (epoch)
        wandb.log(report)

    def print_report(self,
                    args: Optional[argparse.Namespace] = None,
                    reset: bool = True,
                    log_to_wandb: bool = False,
                    current_epoch: Optional[int] = None):  # Added current_epoch parameter

        force_reported_metric: Optional[float] = np.mean(self.force_reported_metrics) if len(self.force_reported_metrics) > 0 else None
        moment_reported_metric: Optional[float] = np.mean(self.moment_reported_metrics) if len(self.moment_reported_metrics) > 0 else None
        cop_reported_metric: Optional[float] = np.mean(self.cop_reported_metrics) if len(self.cop_reported_metrics) > 0 else None
        tau_reported_metric: Optional[float] = np.mean(self.tau_reported_metrics) if len(self.tau_reported_metrics) > 0 else None
        loss_reported_metric: Optional[float] = np.mean(self.losses_reported_metrics) if len(self.losses_reported_metrics) > 0 else None

        if log_to_wandb and len(self.force_losses) > 0:
            assert(args is not None)
            aggregate_force_loss = torch.mean(torch.vstack(self.force_losses), dim=0)
            aggregate_cop_loss = torch.mean(torch.vstack(self.cop_losses), dim=0)
            aggregate_moment_loss = torch.mean(torch.vstack(self.moment_losses), dim=0)
            aggregate_loss = np.mean(np.hstack(self.losses))
            
            # Use current_epoch as the step
            self.log_to_wandb(args,
                              aggregate_force_loss,
                              aggregate_cop_loss,
                              aggregate_moment_loss,
                              aggregate_loss,
                              force_reported_metric,
                              cop_reported_metric,
                              moment_reported_metric,
                              tau_reported_metric,
                              loss_reported_metric,
                              step=current_epoch)
            
        if force_reported_metric is not None:
            print(f'\tForce Avg Err: {force_reported_metric} N / kg')
            print(f'\tCoP Avg Err: {cop_reported_metric} m')
            print(f'\tMoment Avg Err: {moment_reported_metric} Nm / kg')
            print(f'\tNon-root Joint Torques (Inverse Dynamics) Avg Err: {tau_reported_metric} Nm / kg')

        # Reset
        if reset:
            # Aggregating losses across batches for dev set evaluation
            self.losses = []
            self.force_losses = []
            self.moment_losses = []
            self.cop_losses = []

            # Aggregating reported metrics for dev set evaluation
            self.force_reported_metrics = []
            self.moment_reported_metrics = []
            self.cop_reported_metrics = []
            self.tau_reported_metrics = []
            self.tau_reported_metrics_std = []
            self.losses_reported_metrics = []

    def state_dict(self):
        state = {
            'losses': self.losses,
            'force_losses': self.force_losses,
            'moment_losses': self.moment_losses,
            'cop_losses': self.cop_losses,
            'force_reported_metrics': self.force_reported_metrics,
            'moment_reported_metrics': self.moment_reported_metrics,
            'cop_reported_metrics': self.cop_reported_metrics,
            'tau_reported_metrics': self.tau_reported_metrics,
            # 'tau_reported_metrics_std': self.tau_reported_metrics_std,
            'losses_reported_metrics': self.losses_reported_metrics,
        }
        # Serialize the state of GHMR instances if they exist
        state['ghmr_forces'] = [ghmr.state_dict() for ghmr in self.ghmr_forces] if self.ghmr_forces is not None else None
        state['ghmr_moments'] = [ghmr.state_dict() for ghmr in self.ghmr_moments] if self.ghmr_moments is not None else None
        state['ghmr_cops'] = [ghmr.state_dict() for ghmr in self.ghmr_cops] if self.ghmr_cops is not None else None
        return state

    def load_state_dict(self, state_dict):
        self.losses = state_dict['losses']
        self.force_losses = state_dict['force_losses']
        self.moment_losses = state_dict['moment_losses']
        self.cop_losses = state_dict['cop_losses']
        self.force_reported_metrics = state_dict['force_reported_metrics']
        self.moment_reported_metrics = state_dict['moment_reported_metrics']
        self.cop_reported_metrics = state_dict['cop_reported_metrics']
        self.tau_reported_metrics = state_dict['tau_reported_metrics']
        # self.tau_reported_metrics_std = state_dict['tau_reported_metrics_std']
        self.losses_reported_metrics = state_dict['losses_reported_metrics']

        # Load the state of GHMR instances if they exist
        if self.ghmr_forces is not None and state_dict['ghmr_forces'] is not None:
            for ghmr, ghmr_state in zip(self.ghmr_forces, state_dict['ghmr_forces']):
                ghmr.load_state_dict(ghmr_state)
        if self.ghmr_moments is not None and state_dict['ghmr_moments'] is not None:
            for ghmr, ghmr_state in zip(self.ghmr_moments, state_dict['ghmr_moments']):
                ghmr.load_state_dict(ghmr_state)
        if self.ghmr_cops is not None and state_dict['ghmr_cops'] is not None:
            for ghmr, ghmr_state in zip(self.ghmr_cops, state_dict['ghmr_cops']):
                ghmr.load_state_dict(ghmr_state)
