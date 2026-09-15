from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import math
import numpy as np
import torchvision
from einops import rearrange

from .semantic_memory import SemanticProtoConfig, SemanticPrototypeMemory, safe_normalize


def update_ema(ema, new_data):
    if ema is None:
        return new_data
    else:
        with torch.no_grad():
            return 0.9 * ema + (1 - 0.9) * new_data




class MASA(nn.Module):

    def __init__(self, model, optimizer, sigmas, batch_size, steps=1, episodic=False, margin=0.8*math.log(1000), \
        reset_constant_em=0.2, margin_L0 = 0.8 * math.log(1000), weight_reg = 0.5, reweight_threshold = 3.0, weight_tau = 1.2,
        semantic_cfg=None, plpd_threshold=0.2):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "MASA requires at least one adaptation step"
        self.episodic = episodic
        self.batch_size = batch_size

        self.reset_constant_em = reset_constant_em
        self.ema = None  # to record the moving average of model output entropy, as model recovery criteria

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.margin = margin # margin \tau_RE in Eqn. (9)
        self.margin_L0 = margin_L0 # L_0 in Eqn. (9)
        self.weight_reg = weight_reg
        self.reweight_threshold = reweight_threshold
        self.weight_tau = weight_tau
        self.sigma_t = sigmas
        self.plpd_threshold = plpd_threshold
        if semantic_cfg is None:
            semantic_cfg = SemanticProtoConfig(enabled=False)
        self.semantic_cfg = semantic_cfg
        self.semantic_memory = SemanticPrototypeMemory(
            semantic_cfg,
            device=str(next(model.parameters()).device)
        )

        try:
            self.W = model.fc.weight # ResNet
        except:
            self.W = model.head.weight # ViT
        self.W_cpu = self.W.cpu()
        self._refresh_prob_aug()


    def _refresh_prob_aug(self, scale = 0.1):
        with torch.no_grad():
            device = self.W.device
            sigma_t = self.sigma_t.view(1, 1, -1)
            region = sigma_t * self.weight_tau / scale
            sqrt_region = torch.sqrt(region).cpu()
            diff = (self.W_cpu.unsqueeze(0) - self.W_cpu.unsqueeze(1)) * sqrt_region
            self.prob_aug = torch.exp(0.5 * torch.einsum('ijb,ijb->ij', diff, diff))
            self.prob_aug = self.prob_aug.to(device)
            self.normW = 0.1 / 2 * (scale ** 2)  * (torch.norm(self.W, dim=1) ** 2)

    @torch.jit.script
    def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)

    def L_RE(self, x: torch.Tensor) -> torch.Tensor:
        """Implicit augmentation using gaussian noise. speed up"""
        prob_anchor = x.softmax(1)
        prob_aug = (prob_anchor.unsqueeze(1) * self.prob_aug).sum(2)
        prob = (x + self.normW).softmax(1)
        return (-prob * torch.log(prob_anchor) + prob * torch.log(prob_aug)).sum(1)

    def L_RI(self, x: torch.Tensor) -> torch.Tensor:
        prob_anchor = x.softmax(1)
        prob_aug = (prob_anchor.unsqueeze(1) * self.prob_aug).sum(2)
        return (prob_anchor * torch.log(prob_aug)).sum(1)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt_masa(self, x, ema):

        self.optimizer.zero_grad()

        outputs = self.model(x)
        features = safe_normalize(self.get_features(x).float(), dim=1)
        
        L_RE = self.L_RE(outputs)
        L_RI = self.L_RI(outputs)
        reliabilities = [1.0] * outputs.shape[0]

        if self.semantic_cfg.enabled:
            with torch.no_grad():
                reliabilities = self.semantic_memory.push_window(
                    x, features.detach(), outputs.softmax(1).detach(), L_RE.detach(), L_RI.detach()
                )
                self.semantic_memory.maybe_refresh_anchors()

        filter_ids_1 = torch.where(L_RE < self.margin)[0]

        L_RE = L_RE[filter_ids_1]
        L_RI = L_RI[filter_ids_1]

########################################  plpd  ####################################################
        if filter_ids_1.numel() > 0:
            x_prime = x[filter_ids_1]
            x_prime = x_prime.detach()
            
            patch_len = int(4)
            resize_t = torchvision.transforms.Resize(((x.shape[-1]//patch_len)*patch_len,(x.shape[-1]//patch_len)*patch_len))
            resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
            x_prime = resize_t(x_prime)
            x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
            perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1], device=x_prime.device), dim=-1)
            x_prime = x_prime[torch.arange(x_prime.shape[0], device=x_prime.device).unsqueeze(-1),perm_idx]
            x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)
            x_prime = resize_o(x_prime)
            
            with torch.no_grad():
                outputs_prime = self.model(x_prime)
            
            prob_outputs = outputs[filter_ids_1].softmax(1)
            prob_outputs_prime = outputs_prime.softmax(1)

            cls1 = prob_outputs.argmax(dim=1)

            plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
            plpd = plpd.reshape(-1)
            filter_ids_2 = torch.where(plpd > self.plpd_threshold)[0]
        else:
            filter_ids_2 = torch.empty(0, dtype=torch.long, device=outputs.device)
       
########################################  add reweighting coefficient  ####################################################

        L_RE = L_RE[filter_ids_2]
        L_RI = L_RI[filter_ids_2]

        RE = L_RE.detach().clone()
        RI = L_RI.detach().clone()

        if L_RE.numel() > 0:
            coeff = torch.minimum(
                torch.exp(self.margin_L0 - RE),
                torch.tensor(self.reweight_threshold, device=RE.device, dtype=RE.dtype)
            )
            masa_loss = (L_RE + self.weight_reg * L_RI).mul(coeff).mean(0)
        else:
            masa_loss = outputs.sum() * 0.0

        proto_loss = outputs.sum() * 0.0
        assignments = []
        if self.semantic_cfg.enabled:
            proto_loss, assignments = self.semantic_memory.proto_loss(features, outputs, reliabilities)

        if self.semantic_cfg.update_mode == "memory_only":
            loss = masa_loss.detach() + proto_loss.detach()
        else:
            loss = masa_loss + self.semantic_cfg.beta_proto * proto_loss

########################################  add reweighting coefficient  ####################################################

        if not np.isnan(loss.item()):
            ema = update_ema(ema, loss.item() / 2) # record moving average loss values for model recovery
        
        if self.semantic_cfg.update_mode != "memory_only":
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if self.semantic_cfg.enabled:
            with torch.no_grad():
                if not assignments:
                    _, assignments = self.semantic_memory.proto_loss(features.detach(), outputs.detach(), reliabilities)
                self.semantic_memory.update_from_batch(features.detach(), outputs.detach(), reliabilities, assignments)
                outputs = self.semantic_memory.fuse_logits(outputs.detach(), assignments)

        reset_flag = False
        if ema is not None:
            if ema < 0.2:
                print("ema < 0.2, now reset the model")
                reset_flag = True

        
        return outputs, ema, reset_flag


    def forward(self, x, no_adapt = False):
        if self.episodic:
            self.reset()

        if no_adapt == True:
            return self.model(x)

        for _ in range(self.steps):
            outputs, ema, reset_flag = self.forward_and_adapt_masa(x, self.ema)

            if reset_flag:
                self.reset()
            self.ema = ema  # update moving average value of loss

        return outputs

    def get_features(self, x):
        x = self.model.forward_features(x)

        try:
            x = self.model.global_pool(x) # ResNet
        except:
            x = x[:, 0, :]  # ViT

        return x
        
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None
        self.semantic_memory.reset()




def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names



def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with MASA."""
    # train mode, because MASA optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what MASA updates
    model.requires_grad_(False)
    # configure norm for MASA updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

