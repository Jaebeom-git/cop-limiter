import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
from typing import Dict, Callable

"""
Revised SparseMixer implementation based on the SparseMixer: https://github.com/microsoft/SparseMixer.git
"""

class SparseMixerCore(torch.autograd.Function):
    """
    `torch.autograd.Function` implementation of the balancing strategy used in SparseMixer.
    """
    
    @staticmethod
    def forward(
        ctx, 
        multiplier: torch.Tensor, 
        firstorder_mask: torch.Tensor,
    ):
        firstorder_mask = torch.add(0.5, firstorder_mask, alpha=0.5).type_as(multiplier)
        return multiplier * firstorder_mask # turns [0,1] into [0.5, 1]
    
    @staticmethod
    def backward(
        ctx, 
        grad_at_multiplier: torch.Tensor, 
    ):
        return grad_at_multiplier * 2, None
    
class SparseMixer(nn.Module):
    def __init__(self, num_experts, embed_dim, compute_balance_loss=False, jitter_eps=0.1):
        super(SparseMixer, self).__init__()
        self.num_experts = num_experts
        self.compute_balance_loss = compute_balance_loss
        self.jitter_eps = jitter_eps
        self.embed_dim = embed_dim
        self.register_parameter('omega', torch.nn.Parameter(torch.ones(embed_dim)))

    def forward(self, logits):
        
        # masking out experts that are never sampled by jittering 
        with torch.no_grad():
            mask_logits_threshold, max_ind = logits.max(dim=-1, keepdim=True)
            factor = logits.abs().clamp(min=mask_logits_threshold)
            mask_logits_threshold = (
                (mask_logits_threshold - logits) / factor
            ) > (2 * self.jitter_eps)
        logits = logits.masked_fill_(mask_logits_threshold, float('-inf'))
        
        p = logits.softmax(dim=-1)
        if self.training:
            sample = (
                logits - torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
            ).max(dim=-1)[1].unsqueeze(-1) # gumbel sampling, more robust than than the multinomial method
            
            multiplier = p.gather(dim=1, index=sample)
            
            mask_for_firstorder = torch.logical_or(
                sample == max_ind,
                torch.rand_like(multiplier) > 0.5
            ) # compute the mask for applying the first-order method
            multiplier = SparseMixerCore.apply(multiplier, mask_for_firstorder) # balance mid-point and euler 
        else:
            sample = max_ind
            multiplier = p.gather(dim=1, index=sample)
        
        multiplier = multiplier * self.omega 
        balance_loss = 0.0
        if self.compute_balance_loss:
            num_tokens = F.one_hot(sample.squeeze(-1), self.num_experts).gt(0).sum(0)
            f = num_tokens / (num_tokens.sum(0, keepdim=True) + 1e-6)
            pmean = p.view(-1, self.num_experts).mean(0) 
            balance_loss = self.num_experts * torch.sum(pmean * f)
        
        return sample, multiplier, balance_loss
        # return sample, [multiplier], balance_loss


uniform_map: Dict[torch.device, Callable] = {}
def multiplicative_jitter(input, epsilon, training):

    if epsilon == 0 or not training:
        return input

    uniform = uniform_map.get(input.device)

    if uniform is None:
        uniform = Uniform(low=torch.tensor(1.0 - epsilon, device=input.device, dtype=input.dtype),
                          high=torch.tensor(1.0 + epsilon, device=input.device, dtype=input.dtype)
                ).rsample
        uniform_map[input.device] = uniform

    return input * uniform(input.shape)

class v2core(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        scores: torch.Tensor, 
        multiplier: torch.Tensor, 
        selected_experts: torch.Tensor,
        masked_gates: torch.Tensor,
        mask_for_one: torch.Tensor,
    ):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one
        
    @staticmethod
    def backward(
        ctx, 
        grad_at_output: torch.Tensor, 
    ):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors
        
        grad_at_output = grad_at_output * multiplier
        
        grad_at_scores_expaned = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expaned.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )
        
        return (
            grad_at_scores_expaned, 
            None, 
            None, 
            None, 
            None, 
        )

def sparsemixerv2_routing(scores, top_k, jitter_eps, training):
    assert top_k in [1, 2], "only top-1/2 gating has been tested!"
    
    original_gates = torch.softmax(scores, dim=-1)
    ################ first expert ################
    
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = (
            (mask_logits_threshold - scores) / factor
        ) > (2 * jitter_eps)

    # apply mask 
    masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
    if training:
        selected_experts = (
            masked_gates - torch.empty_like(masked_gates, memory_format=torch.legacy_contiguous_format).exponential_().log()
        ).max(dim=-1)[1].unsqueeze(-1) # gumbel sampling, more robust than than the multinomial method
    else:
        selected_experts = max_ind
        
    # compute scores for gradients
    masked_gates = torch.softmax(masked_gates, dim=-1)
    
    # compute midpoint mask 
    max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
    mask_for_one = torch.logical_or(
        selected_experts == max_ind,
        torch.rand_like(max_scores) > 0.75 # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
    ) 
    # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
    mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)
    multiplier = v2core.apply(
        scores, 
        multiplier_o, 
        selected_experts, 
        masked_gates, 
        mask_for_one,
    )
    
    ################ second expert ################
    if top_k > 1:
        # masked out first expert 
        masked_scores = torch.scatter(
            scores,
            -1,
            selected_experts,
            float('-inf'),
        )
        with torch.no_grad():
            # compute mask for sparsity
            mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
            factor = scores.abs().clamp(min=mask_logits_threshold)
            mask_logits_threshold = (
                (mask_logits_threshold - scores) / factor
            ) > (2 * jitter_eps)

        # apply mask 
        masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float('-inf'))
        if training:
            selected_experts_top2 = (
                masked_gates_top2 - torch.empty_like(masked_gates_top2, memory_format=torch.legacy_contiguous_format).exponential_().log()
            ).max(dim=-1)[1].unsqueeze(-1) # gumbel sampling, more robust than than the multinomial method
        else:
            selected_experts_top2 = max_ind
        # compute scores for gradients
        masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
        
        # compute midpoint mask 
        max_scores, max_ind = masked_gates_top2.max(dim=-1, keepdim=True)
        mask_for_one_top2 = torch.logical_or(
            selected_experts_top2 == max_ind,
            torch.rand_like(max_scores).uniform_() > 0.75 # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
        ) 
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one_top2 = torch.add(0.3333, mask_for_one_top2, alpha=0.6667).type_as(masked_gates_top2)

        multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)
        multiplier_top2 = v2core.apply(
            scores, 
            multiplier_top2_o, 
            selected_experts_top2, 
            masked_gates_top2, 
            mask_for_one_top2,
        )
        
        multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
        selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)
            
    return (
        multiplier, 
        original_gates, 
        selected_experts,
    )

class SparseMixerV2(SparseMixer):
    def __init__(self, num_experts, embed_dim, compute_balance_loss=False, jitter_eps=0.1, top_k=2):
        super(SparseMixerV2, self).__init__(num_experts, embed_dim, compute_balance_loss, jitter_eps)
        self.top_k = top_k

    def forward(self, logits):
        
        multiplier, original_gates, sample = sparsemixerv2_routing(logits, top_k=self.top_k, jitter_eps=self.jitter_eps, training=self.training)

        balance_loss = 0.0
        if self.compute_balance_loss:
            num_tokens = F.one_hot(sample.squeeze(-1), self.num_experts).gt(0).sum(0)
            f = num_tokens / (num_tokens.sum(0, keepdim=True) + 1e-6)
            pmean = original_gates.view(-1, self.num_experts).mean(0)
            balance_loss = self.num_experts * torch.sum(pmean * f)
        
            return sample, multiplier, balance_loss
        
        else:
            return sample, multiplier, original_gates
