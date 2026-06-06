import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import List, Dict, Tuple

# Fix OpenMP duplicate library error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ============================================================
# Utilities
# ============================================================

def compute_stable_rank(W: torch.Tensor) -> float:
    """Computes the stable rank of a matrix W"""
    if W.dim() != 2:
        raise ValueError("Stable rank is only defined for 2D matrices.")
    
    frob_norm_sq = torch.norm(W, p='fro')**2
    
    # Use SVD for exact operator norm (largest singular value)
    _, S, _ = torch.svd(W)
    op_norm_sq = S[0]**2
    
    if op_norm_sq < 1e-10:
        return 0.0
        
    return (frob_norm_sq / op_norm_sq).item()

def generate_target_matrix(dim: int, rank: int, sr_multiplier: float, device: str) -> torch.Tensor:
    """
    Generates a matrix with a specific rank and a controlled stable rank profile.
    sr_multiplier controls the magnitude of the first singular value relative to the rest.
    Higher multiplier -> lower stable rank.
    """
    U = torch.randn(dim, rank, device=device)
    V = torch.randn(rank, dim, device=device)
    
    # Orthogonalize U and V to control singular values exactly
    U, _ = torch.linalg.qr(U)
    V, _ = torch.linalg.qr(V.T)
    V = V.T
    
    # Flat singular values
    S = torch.ones(rank, device=device) + torch.randn(rank, device=device) * 0.1
    
    # Modify the first singular value
    S[0] = sr_multiplier
        
    # Scale S so the Frobenius norm is consistent (e.g., sqrt(rank))
    # This ensures the "total energy" of the shift is the same across different SRs
    # S = S / torch.norm(S) * np.sqrt(rank)

    S = S / torch.norm(S)
    
    return U @ torch.diag(S) @ V

# ============================================================
# Model Definitions
# ============================================================

class MLP4(nn.Module):
    """4-layer MLP with SiLU activations."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),  # Target for LoRA 1
            nn.Linear(hidden_dim, hidden_dim),  # Target for LoRA 2
            nn.Linear(hidden_dim, output_dim)
        ])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.layers[0](x))
        h = self.act(self.layers[1](h))
        h = self.act(self.layers[2](h))
        return self.layers[3](h)

def make_mlp(n_layers: int, input_dim: int, hidden_dim: int, output_dim: int = None) -> nn.Module:
    """Creates an MLP with n_layers linear layers and SiLU activations between all but the last.
    First layer: input_dim → hidden_dim. Last: hidden_dim → output_dim. Middle: hidden_dim → hidden_dim.
    If output_dim is None, it defaults to input_dim."""
    if output_dim is None:
        output_dim = input_dim
        
    class _FlexMLP(nn.Module):
        def __init__(self):
            super().__init__()
            layer_list = []
            for i in range(n_layers):
                if n_layers == 1:
                    # Edge case: single layer network
                    layer_list.append(nn.Linear(input_dim, output_dim))
                elif i == 0:
                    layer_list.append(nn.Linear(input_dim, hidden_dim))
                elif i == n_layers - 1:
                    layer_list.append(nn.Linear(hidden_dim, output_dim))
                else:
                    layer_list.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers = nn.ModuleList(layer_list)
            self.act = nn.SiLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x
            for i, layer in enumerate(self.layers):
                h = layer(h)
                if i < len(self.layers) - 1:
                    h = self.act(h)
            return h

    return _FlexMLP()

class LoRALayer(nn.Module):
    """Standard LoRA layer."""
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        self.lora_A = nn.Parameter(torch.empty(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        
        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out
        
    def get_delta_W(self) -> torch.Tensor:
        return (self.lora_A @ self.lora_B).T * self.scaling
        
    def get_stable_rank(self) -> float:
        return compute_stable_rank(self.get_delta_W())

class LoRAMLP4Wrapper(nn.Module):
    """Wraps a 4-layer MLP with LoRA on the two middle layers."""
    def __init__(self, pretrained_mlp: MLP4, rank: int, alpha: float = None):
        super().__init__()
        self.model = pretrained_mlp
        
        # Freeze base model
        for p in self.model.parameters():
            p.requires_grad = False
            
        # Inject LoRA into layers 1 and 2
        self.lora1 = LoRALayer(self.model.layers[1], rank=rank, alpha=alpha if alpha else rank * 2)
        self.lora2 = LoRALayer(self.model.layers[2], rank=rank, alpha=alpha if alpha else rank * 2)
        
        self.model.layers[1] = self.lora1
        self.model.layers[2] = self.lora2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_avg_sr(self) -> float:
        return (self.lora1.get_stable_rank() + self.lora2.get_stable_rank()) / 2.0


class LoRAFlexWrapper(nn.Module):
    """Wraps an MLP with LoRA on a configurable list of layer indices."""
    def __init__(self, model: nn.Module, rank: int, adapter_layers: List[int], alpha: float = None):
        super().__init__()
        self.model = model

        # Freeze base model
        for p in self.model.parameters():
            p.requires_grad = False

        # Inject LoRA into specified layers and store references
        self._lora_layers = nn.ModuleList()
        for idx in adapter_layers:
            lora = LoRALayer(self.model.layers[idx], rank=rank, alpha=alpha if alpha else rank * 2)
            self.model.layers[idx] = lora
            self._lora_layers.append(lora)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_avg_sr(self) -> float:
        srs = [l.get_stable_rank() for l in self._lora_layers]
        return sum(srs) / len(srs) if srs else 0.0


# ============================================================
# Experiment A: The Implicit Low-SR Bias (Expressivity)
# ============================================================

def run_exp_a_expressivity(
    input_dim: int = 64,
    hidden_dim: int = 256,
    ranks: List[int] = [4, 16, 64, 256],
    batch_size: int = 512,
    steps: int = 5000,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "results/toy_exp_a_4layer",
):
    print(f"\n{'='*60}")
    print(f"Experiment A: The Implicit Low-SR Bias (4-Layer)")
    print(f"{'='*60}")
    os.makedirs(save_dir, exist_ok=True)
    
    torch.manual_seed(42)
    teacher = MLP4(input_dim, hidden_dim, input_dim).to(device)
    student_base = MLP4(input_dim, hidden_dim, input_dim).to(device)
    student_base.load_state_dict(teacher.state_dict())
    
    # Apply high-SR shift to BOTH middle layers, with higher magnitude
    shift1 = generate_target_matrix(hidden_dim, hidden_dim, 1.0, device) * 5.0
    shift2 = generate_target_matrix(hidden_dim, hidden_dim, 1.0, device) * 5.0
    with torch.no_grad():
        teacher.layers[1].weight.add_(shift1)
        teacher.layers[2].weight.add_(shift2)
    
    target_sr1 = compute_stable_rank(shift1)
    target_sr2 = compute_stable_rank(shift2)
    print(f"Target Shift SRs: L1={target_sr1:.2f}, L2={target_sr2:.2f}")
    
    def get_batch():
        x = torch.randn(batch_size, input_dim, device=device)
        with torch.no_grad():
            y = teacher(x)
        return x, y

    loss_fn = nn.MSELoss()
    results = {}
    
    # Full FT Baseline
    print("\nTraining Full FT Baseline...")
    full_ft_model = MLP4(input_dim, hidden_dim, input_dim).to(device)
    full_ft_model.load_state_dict(student_base.state_dict())
    
    for name, param in full_ft_model.named_parameters():
        if 'layers.1' in name or 'layers.2' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    opt_ft = optim.AdamW(filter(lambda p: p.requires_grad, full_ft_model.parameters()), lr=lr)
    initial_w1 = full_ft_model.layers[1].weight.data.clone()
    initial_w2 = full_ft_model.layers[2].weight.data.clone()
    
    ft_losses, ft_srs = [], []
    for step in range(steps):
        x, y = get_batch()
        pred = full_ft_model(x)
        loss = loss_fn(pred, y)
        
        opt_ft.zero_grad()
        loss.backward()
        opt_ft.step()
        
        if step % 100 == 0 or step == steps - 1:
            with torch.no_grad():
                sr1 = compute_stable_rank(full_ft_model.layers[1].weight.data - initial_w1)
                sr2 = compute_stable_rank(full_ft_model.layers[2].weight.data - initial_w2)
                avg_sr = (sr1 + sr2) / 2
                ft_losses.append(loss.item())
                ft_srs.append(avg_sr)
            if step % 500 == 0:
                print(f"  Step {step:4d} | Loss: {loss.item():.6f} | Avg SR: {avg_sr:.2f}")
                
    results['full_ft'] = {'losses': ft_losses, 'srs': ft_srs, 'final_loss': ft_losses[-1], 'final_sr': ft_srs[-1]}

    # LoRA models
    for rank in ranks:
        print(f"\nTraining LoRA Rank {rank}...")
        lora_model = LoRAMLP4Wrapper(MLP4(input_dim, hidden_dim, input_dim).to(device), rank=rank).to(device)
        lora_model.model.load_state_dict(student_base.state_dict(), strict=False)
        
        opt_lora = optim.AdamW(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=lr)
        
        lora_losses, lora_srs = [], []
        for step in range(steps):
            x, y = get_batch()
            pred = lora_model(x)
            loss = loss_fn(pred, y)
            
            opt_lora.zero_grad()
            loss.backward()
            opt_lora.step()
            
            if step % 100 == 0 or step == steps - 1:
                sr = lora_model.get_avg_sr()
                lora_losses.append(loss.item())
                lora_srs.append(sr)
                if step % 500 == 0:
                    print(f"  Step {step:4d} | Loss: {loss.item():.6f} | Avg SR: {sr:.2f}")
                    
        results[f'lora_r{rank}'] = {'losses': lora_losses, 'srs': lora_srs, 'final_loss': lora_losses[-1], 'final_sr': lora_srs[-1]}

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for k, v in results.items():
        plt.plot(np.arange(len(v['losses']))*100, v['losses'], label=k)
    plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss (Infinite Data)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for k, v in results.items():
        plt.plot(np.arange(len(v['srs']))*100, v['srs'], label=k)
    plt.axhline(y=(target_sr1+target_sr2)/2, color='r', linestyle='--', label='Target Avg SR')
    plt.xlabel('Steps')
    plt.ylabel('Average Stable Rank')
    plt.title('Stable Rank Evolution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp_a_results.png'))
    plt.close()
    print(f"\nResults saved to {save_dir}")

# ============================================================
# Experiment B: Target SR vs. Learnability
# ============================================================

def run_exp_b_target_sr(
    input_dim: int = 64,
    hidden_dim: int = 256,
    target_rank: int = 256,
    lora_rank: int = 256,
    batch_size: int = 512,
    steps: int = 10000,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "results/toy_exp_b_4layer",
):
    print(f"\n{'='*60}")
    print(f"Experiment B: Target SR vs. Learnability (4-Layer)")
    print(f"{'='*60}")
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(42)
    student_base = MLP4(input_dim, hidden_dim, input_dim).to(device)
    loss_fn = nn.MSELoss()
    results = {}
    
    # Multipliers for the first singular value to control SR
    # 1.0 -> SR ~ 256
    # 20.0 -> SR ~ 1.6
    sr_multipliers = [1.0, 2.0, 5.0, 10.0, 20.0]
    
    for multiplier in sr_multipliers:
        print(f"\nTraining on Target with SR Multiplier {multiplier}...")
        
        teacher = MLP4(input_dim, hidden_dim, input_dim).to(device)
        teacher.load_state_dict(student_base.state_dict())
        
        shift1 = generate_target_matrix(hidden_dim, target_rank, multiplier, device)
        shift2 = generate_target_matrix(hidden_dim, target_rank, multiplier, device)
        with torch.no_grad():
            teacher.layers[1].weight.add_(shift1)
            teacher.layers[2].weight.add_(shift2)
            
        target_sr = (compute_stable_rank(shift1) + compute_stable_rank(shift2)) / 2
        print(f"  Target True Rank: {target_rank}, Target Avg Stable Rank: {target_sr:.2f}")
        
        def get_batch():
            x = torch.randn(batch_size, input_dim, device=device)
            with torch.no_grad():
                y = teacher(x)
            return x, y
            
        # Train Full SFT
        print(f"  Training Full SFT...")
        full_ft_model = MLP4(input_dim, hidden_dim, input_dim).to(device)
        full_ft_model.load_state_dict(student_base.state_dict())
        # Store initial weights for SR calculation
        initial_w1 = full_ft_model.layers[1].weight.data.clone()
        initial_w2 = full_ft_model.layers[2].weight.data.clone()
        for name, param in full_ft_model.named_parameters():
            param.requires_grad = 'layers.1' in name or 'layers.2' in name
        opt_ft = optim.AdamW(filter(lambda p: p.requires_grad, full_ft_model.parameters()), lr=lr)
        
        ft_losses = []
        for step in range(steps):
            x, y = get_batch()
            pred = full_ft_model(x)
            loss = loss_fn(pred, y)
            
            opt_ft.zero_grad()
            loss.backward()
            opt_ft.step()
            
            if step % 100 == 0 or step == steps - 1:
                ft_losses.append(loss.item())
        
        # Calculate stable rank for full fine-tuning
        with torch.no_grad():
            full_ft_sr1 = compute_stable_rank(full_ft_model.layers[1].weight.data - initial_w1)
            full_ft_sr2 = compute_stable_rank(full_ft_model.layers[2].weight.data - initial_w2)
            full_ft_sr = (full_ft_sr1 + full_ft_sr2) / 2
            print(f"  Full FT final SR: {full_ft_sr:.2f}")
                
        # Train LoRA
        print(f"  Training LoRA...")
        lora_model = LoRAMLP4Wrapper(MLP4(input_dim, hidden_dim, input_dim).to(device), rank=lora_rank).to(device)
        lora_model.model.load_state_dict(student_base.state_dict(), strict=False)
        opt_lora = optim.AdamW(filter(lambda p: p.requires_grad, lora_model.parameters()), lr=lr)
        
        lora_losses = []
        for step in range(steps):
            x, y = get_batch()
            pred = lora_model(x)
            loss = loss_fn(pred, y)
            
            opt_lora.zero_grad()
            loss.backward()
            opt_lora.step()
            
            if step % 100 == 0 or step == steps - 1:
                lora_losses.append(loss.item())
        
        # Get LoRA stable rank
        lora_sr = lora_model.get_avg_sr()
        print(f"  LoRA final SR: {lora_sr:.2f}")
                    
        results[f'sr_{target_sr:.1f}'] = {
            'lora_losses': lora_losses,
            'ft_losses': ft_losses,
            'target_sr': target_sr,
            'multiplier': multiplier,
            'lora_sr': lora_sr,
            'full_ft_sr': full_ft_sr
        }

    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    plt.figure(figsize=(12, 6))
    
    # Plot LoRA
    plt.subplot(1, 2, 1)
    for k, v in results.items():
        plt.plot(np.arange(len(v['lora_losses']))*100, v['lora_losses'], label=f"Target SR: {v['target_sr']:.1f}")
    plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('MSE Loss')
    plt.title(f'LoRA (Rank {lora_rank})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Full SFT
    plt.subplot(1, 2, 2)
    for k, v in results.items():
        plt.plot(np.arange(len(v['ft_losses']))*100, v['ft_losses'], label=f"Target SR: {v['target_sr']:.1f}")
    plt.yscale('log')
    plt.xlabel('Steps')
    plt.ylabel('MSE Loss')
    plt.title(f'Full SFT')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Learning Different SR Targets (Same Frobenius Norm)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'exp_b_results.png'))
    plt.close()
    print(f"\nResults saved to {save_dir}")

# ============================================================
# Experiment C: Generalization Gap vs. Stable Rank
# ============================================================

def run_exp_c_generalization(
    input_dim: int = 128,
    hidden_dim: int = 128,
    ranks: List[int] = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
    conditions: dict = None,         # sr_multiplier per condition label; default below
    train_samples: int = 2048,
    val_samples: int = 10240,
    noise_std: float = 0.15,
    batch_size: int = 32,
    epochs: int = 200,
    lr: float = 1e-3,
    n_seeds: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_layers: int = 10,
    shift_layers: List[int] = [1,2,3,4,5,6,7,8,9,],
    adapter_layers: List[int] =  [1,2,3,4,5,6,7,8,9,],
    shift_multiplier: float = 0.5,
    save_dir: str = "results/toy_exp_c/sr_not_growing/n_train_2048_with_full_FT",
):
    """
    Experiment C: SR as a Universal Predictor of Generalization.

    Two teacher conditions differ only in the stable rank of the target shift
    applied to layers 1 and 2 of the MLP:

      - "complex": sr_multiplier=1.0  → high-SR target shift (SR ≈ hidden_dim)
      - "simple":  sr_multiplier=20.0 → low-SR target shift  (SR ≈ 1–2)

    Standard LoRA is trained at many nominal ranks across both conditions.  The
    key finding is that the generalization gap is predicted by the *trained*
    adapter SR regardless of which condition the data came from (pooled scatter
    falls on a single line), while nominal rank alone gives two separate curves
    — one per condition.
    """
    if conditions is None:
        conditions = {
            "c1": 1,
            "c2": 1.5,
            "c3": 2,
            "c4": 3.0,
            "c5": 5.0,
            "c6": 8.0,
            "c7": 12.0,
            "c8": 20.0,
        }

    print(f"\n{'='*60}")
    print(f"Experiment C: SR as Universal Generalization Predictor (4-Layer)")
    print(f"  conditions={conditions}, ranks={ranks}, n_seeds={n_seeds}")
    print(f"  train_samples={train_samples}, val_samples={val_samples}, "
          f"noise_std={noise_std}, epochs={epochs}")
    print(f"{'='*60}")
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Deterministic teacher base (seed 0); teachers are built per condition
    # inside the seed loop so each condition gets the same base weights.
    # ------------------------------------------------------------------
    torch.manual_seed(0)
    student_base = make_mlp(n_layers, input_dim, hidden_dim).to(device)

    mse_loss_fn = nn.MSELoss()

    # ------------------------------------------------------------------
    # Measure and log the target SR for each condition (using seed-0 draws)
    # ------------------------------------------------------------------
    target_srs: Dict[str, float] = {}
    for label, sr_mult in conditions.items():
        torch.manual_seed(0)
        shifts = [generate_target_matrix(hidden_dim, hidden_dim, sr_mult, device) * shift_multiplier
                  for _ in shift_layers]
        avg_target_sr = sum(compute_stable_rank(s) for s in shifts) / len(shifts)
        target_srs[label] = float(avg_target_sr)
        print(f"  Condition '{label}' (sr_mult={sr_mult}): "
              f"target shift avg SR = {avg_target_sr:.2f}")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Outer sweep: seeds → conditions → ranks
    # ------------------------------------------------------------------
    results: Dict[str, Dict] = {}

    for seed in range(n_seeds):
        torch.manual_seed(seed + 100)

        for label, sr_mult in conditions.items():
            # ---- Build teacher for this condition ----
            teacher_cond = copy.deepcopy(student_base)
            shifts = [generate_target_matrix(hidden_dim, hidden_dim, sr_mult, device) * shift_multiplier
                      for _ in shift_layers]
            with torch.no_grad():
                for idx, s in zip(shift_layers, shifts):
                    teacher_cond.layers[idx].weight.data.add_(s * ((teacher_cond.layers[idx].weight ** 2).mean()**0.5))

            measured_sr = sum(compute_stable_rank(s) for s in shifts) / len(shifts)
            print(f"\n[seed={seed}  cond='{label}'  sr_mult={sr_mult}  "
                  f"measured_target_SR={measured_sr:.2f}]")
            sys.stdout.flush()

            # ---- Generate data for this (condition, seed) ----
            x_train = torch.randn(train_samples, input_dim, device=device)
            with torch.no_grad():
                y_train = (teacher_cond(x_train).detach()
                           + noise_std * torch.randn(train_samples, input_dim, device=device))

            x_val = torch.randn(val_samples, input_dim, device=device)
            with torch.no_grad():
                y_val = (teacher_cond(x_val).detach()
                         + noise_std * torch.randn(val_samples, input_dim, device=device))

            # ---- Train full fine-tuning baseline ----
            run_key = f"{label}_full_ft_seed{seed}"
            print(f"\n  --- Run: {run_key} (Full Fine-Tuning) ---")
            sys.stdout.flush()
            
            # Create a fresh copy of the student base model
            fresh_mlp_ft = copy.deepcopy(student_base)
            
            # Store initial weights for SR calculation
            initial_weights = {}
            for idx in adapter_layers:
                initial_weights[idx] = fresh_mlp_ft.layers[idx].weight.data.clone()
            
            # Freeze all parameters except those in adapter_layers
            for name, param in fresh_mlp_ft.named_parameters():
                param.requires_grad = False
            
            for idx in adapter_layers:
                fresh_mlp_ft.layers[idx].weight.requires_grad = True
                if fresh_mlp_ft.layers[idx].bias is not None:
                    fresh_mlp_ft.layers[idx].bias.requires_grad = True
            
            # Use the same optimizer settings
            optimizer_ft = optim.AdamW(
                [p for p in fresh_mlp_ft.parameters() if p.requires_grad],
                lr=lr, weight_decay=0.0,
            )
            
            # Train with the same settings
            N = x_train.size(0)
            for epoch in range(epochs):
                fresh_mlp_ft.train()
                perm = torch.randperm(N, device=device)
                for i in range(0, N, batch_size):
                    idx = perm[i: min(i + batch_size, N)]
                    pred = fresh_mlp_ft(x_train[idx])
                    loss = mse_loss_fn(pred, y_train[idx])
                    optimizer_ft.zero_grad()
                    loss.backward()
                    optimizer_ft.step()
                
                if epoch % 1000 == 0:
                    fresh_mlp_ft.eval()
                    with torch.no_grad():
                        full_train_mse = mse_loss_fn(fresh_mlp_ft(x_train), y_train).item()
                    print(f"    Epoch {epoch:5d} | train MSE: {full_train_mse:.6f}")
                    sys.stdout.flush()
            
            # Calculate final metrics
            fresh_mlp_ft.eval()
            with torch.no_grad():
                train_loss_ft = mse_loss_fn(fresh_mlp_ft(x_train), y_train).item()
                val_loss_ft = mse_loss_fn(fresh_mlp_ft(x_val), y_val).item()
            
            # Calculate stable rank of weight differences
            srs = []
            with torch.no_grad():
                for idx in adapter_layers:
                    weight_diff = fresh_mlp_ft.layers[idx].weight.data - initial_weights[idx]
                    sr = compute_stable_rank(weight_diff)
                    srs.append(sr)
            
            avg_sr_ft = sum(srs) / len(srs)
            gap_ft = val_loss_ft - train_loss_ft
            
            # Record results
            results[run_key] = {
                'condition': label,
                'rank': 'full_ft',  # Special marker for full fine-tuning
                'seed': int(seed),
                'train_loss': float(train_loss_ft),
                'val_loss': float(val_loss_ft),
                'gap': float(gap_ft),
                'avg_sr': float(avg_sr_ft),
            }
            
            print(f"  [{run_key}] train={train_loss_ft:.5f} | "
                  f"val={val_loss_ft:.5f} | gap={gap_ft:+.5f} | SR={avg_sr_ft:.3f}")
            sys.stdout.flush()
            
            del fresh_mlp_ft
            
            # ---- Train LoRA at each rank ----
            for r in ranks:
                run_key = f"{label}_r{r}_seed{seed}"
                print(f"\n  --- Run: {run_key} ---")
                sys.stdout.flush()

                fresh_mlp = copy.deepcopy(student_base)
                student = LoRAFlexWrapper(fresh_mlp, rank=r, adapter_layers=adapter_layers).to(device)

                optimizer = optim.AdamW(
                    [p for p in student.parameters() if p.requires_grad],
                    lr=lr, weight_decay=0.0,
                )

                N = x_train.size(0)
                for epoch in range(epochs):
                    student.train()
                    perm = torch.randperm(N, device=device)
                    for i in range(0, N, batch_size):
                        idx = perm[i: min(i + batch_size, N)]
                        pred = student(x_train[idx])
                        loss = mse_loss_fn(pred, y_train[idx])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    if epoch % 1000 == 0:
                        student.eval()
                        with torch.no_grad():
                            full_train_mse = mse_loss_fn(student(x_train), y_train).item()
                        avg_sr_log = student.get_avg_sr()
                        print(f"    Epoch {epoch:5d} | train MSE: {full_train_mse:.6f} "
                              f"| avg SR: {avg_sr_log:.3f}")
                        sys.stdout.flush()

                student.eval()
                with torch.no_grad():
                    train_loss = mse_loss_fn(student(x_train), y_train).item()
                    val_loss = mse_loss_fn(student(x_val), y_val).item()

                avg_sr = student.get_avg_sr()
                gap = val_loss - train_loss

                results[run_key] = {
                    'condition': label,
                    'rank': int(r),
                    'seed': int(seed),
                    'train_loss': float(train_loss),
                    'val_loss': float(val_loss),
                    'gap': float(gap),
                    'avg_sr': float(avg_sr),
                }

                print(f"  [{run_key}] train={train_loss:.5f} | "
                      f"val={val_loss:.5f} | gap={gap:+.5f} | SR={avg_sr:.3f}")
                sys.stdout.flush()

                del student, fresh_mlp

            del teacher_cond

    # ------------------------------------------------------------------
    # Save results + metadata
    # ------------------------------------------------------------------
    output = {
        'metadata': {
            'conditions': conditions,
            'target_srs': target_srs,
            'noise_std': noise_std,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'batch_size': batch_size,
            'epochs': epochs,
            'n_seeds': n_seeds,
            'ranks': ranks,
            'hidden_dim': hidden_dim,
            'input_dim': input_dim,
        },
        'results': results,
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------
    ranks_arr = np.array(ranks, dtype=float)

    def _agg(cond_label, rank, field):
        if rank == 'full_ft':
            vals = [results[f"{cond_label}_full_ft_seed{s}"][field]
                    for s in range(n_seeds)]
        else:
            vals = [results[f"{cond_label}_r{rank}_seed{s}"][field]
                    for s in range(n_seeds)]
        return np.array(vals, dtype=float)

    # Colour / marker per condition
    cond_labels = list(conditions.keys())
    cond_colors  = {'complex': 'tab:red',  'simple': 'tab:blue'}
    cond_markers = {'complex': 's',         'simple': 'o'}
    default_colors  = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown']
    default_markers = ['^', 'D', 'v', 'P']
    for i, lbl in enumerate(cond_labels):
        if lbl not in cond_colors:
            cond_colors[lbl]  = default_colors[i % len(default_colors)]
            cond_markers[lbl] = default_markers[i % len(default_markers)]

    # ------------------------------------------------------------------
    # Figure: three panels
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # ---- Panel 1: Trained SR vs rank ----
    for lbl in cond_labels:
        color = cond_colors[lbl]
        # LoRA points
        sr_mean = np.array([_agg(lbl, r, 'avg_sr').mean() for r in ranks])
        sr_std  = np.array([_agg(lbl, r, 'avg_sr').std()  for r in ranks])
        ax1.errorbar(ranks_arr, sr_mean, yerr=sr_std,
                     color=color, marker='o', capsize=3,
                     label=f"{lbl} (target SR≈{target_srs[lbl]:.1f})")
        
        # Full FT horizontal line
        try:
            ft_sr_mean = _agg(lbl, 'full_ft', 'avg_sr').mean()
            ax1.axhline(y=ft_sr_mean, color=color, linestyle='--', alpha=0.7)
        except:
            # Skip if no full FT data for this condition
            pass

    ax1.set_xlabel('Nominal rank')
    ax1.set_ylabel('Trained adapter avg SR')
    ax1.set_title('Trained adapter SR vs nominal rank')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # ---- Panel 2: Generalization gap vs rank ----
    for lbl in cond_labels:
        color = cond_colors[lbl]
        gap_mean = np.array([_agg(lbl, r, 'gap').mean() for r in ranks])
        gap_std  = np.array([_agg(lbl, r, 'gap').std()  for r in ranks])
        ax2.errorbar(ranks_arr, gap_mean, yerr=gap_std,
                     color=color, marker='o', capsize=3,
                     label=lbl)
        
        # Full FT horizontal line
        try:
            ft_gap_mean = _agg(lbl, 'full_ft', 'gap').mean()
            ax2.axhline(y=ft_gap_mean, color=color, linestyle='--', alpha=0.7)
        except:
            # Skip if no full FT data for this condition
            pass

    ax2.axhline(y=noise_std ** 2, color='grey', linestyle='--',
                alpha=0.7, label=f'noise var = {noise_std**2:.4f}')
    ax2.set_xlabel('Nominal rank')
    ax2.set_ylabel('Generalization gap (val − train)')
    ax2.set_title('Generalization gap (val \u2212 train) vs nominal rank')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # ---- Panel 3: Gap vs √(trained SR) — one line per condition ----
    for lbl in cond_labels:
        color = cond_colors[lbl]
        avg_sr_vals  = np.array([_agg(lbl, r, 'avg_sr').mean() for r in ranks])
        gap_vals     = np.array([_agg(lbl, r, 'gap').mean()    for r in ranks])
        sqrt_sr_vals = np.sqrt(np.maximum(avg_sr_vals, 0.0))
        sort_idx     = np.argsort(sqrt_sr_vals)
        sorted_sqrt_sr = sqrt_sr_vals[sort_idx]
        sorted_gap     = gap_vals[sort_idx]
        ax3.plot(sorted_sqrt_sr, sorted_gap, marker='o', color=color, label=lbl)
        
        # Add Full FT point to the scatter plot
        try:
            ft_sr = _agg(lbl, 'full_ft', 'avg_sr').mean()
            ft_gap = _agg(lbl, 'full_ft', 'gap').mean()
            sqrt_ft_sr = np.sqrt(max(ft_sr, 0.0))
            ax3.plot(sqrt_ft_sr, ft_gap, '*', color=color, markersize=12, label=f"{lbl} Full FT")
        except:
            # Skip if no full FT data for this condition
            pass

    ax3.set_xlabel(r'$\sqrt{\mathrm{SR}(\Delta W)}$')
    ax3.set_ylabel('Generalization gap (val \u2212 train)')
    ax3.set_title('Generalization gap vs \u221a(trained SR)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # ---- Panel 4: Gap vs nominal rank ----
    for lbl in cond_labels:
        color    = cond_colors[lbl]
        gap_mean = np.array([_agg(lbl, r, 'gap').mean() for r in ranks])
        ax4.plot((ranks_arr) ** 0.5, gap_mean, marker='o', color=color, label=lbl)
        
        # Add Full FT horizontal line
        try:
            ft_gap_mean = _agg(lbl, 'full_ft', 'gap').mean()
            ax4.axhline(y=ft_gap_mean, color=color, linestyle='--', alpha=0.7)
        except:
            # Skip if no full FT data for this condition
            pass

    ax4.set_xlabel(r'$\sqrt{rank}$')
    ax4.set_ylabel('Generalization gap (val \u2212 train)')
    ax4.set_title('Generalization gap vs \u221a(rank)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    fig.suptitle(
        f"Exp C  |  noise_std={noise_std}, N_train={train_samples}, epochs={epochs}"
        f" | n_layers={n_layers} | shift={shift_layers} | adapter={adapter_layers}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(save_dir, 'exp_c_results.png'), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Stdout summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("Exp C summary (mean \u00b1 std over seeds)")
    print("=" * 68)
    header = (f"{'condition':>9} | {'rank':>4} | "
              f"{'SR (mean\u00b1std)':>16} | "
              f"{'gap (mean\u00b1std)':>22}")
    print(header)
    print("-" * len(header))
    for lbl in cond_labels:
        # First print full fine-tuning results
        try:
            sr_v_ft = _agg(lbl, 'full_ft', 'avg_sr')
            gp_v_ft = _agg(lbl, 'full_ft', 'gap')
            print(f"{lbl:>9} | {'Full':>4} | "
                  f"{sr_v_ft.mean():>7.2f}\u00b1{sr_v_ft.std():<7.2f} | "
                  f"{gp_v_ft.mean():>+10.3E}\u00b1{gp_v_ft.std():<10.3E}")
        except:
            # Skip if no full FT data for this condition
            pass
            
        # Then print LoRA results
        for r in ranks:
            sr_v  = _agg(lbl, r, 'avg_sr')
            gp_v  = _agg(lbl, r, 'gap')
            print(f"{lbl:>9} | {r:>4d} | "
                  f"{sr_v.mean():>7.2f}\u00b1{sr_v.std():<7.2f} | "
                  f"{gp_v.mean():>+10.3E}\u00b1{gp_v.std():<10.3E}")
    print("=" * 68)
    print(f"\nResults saved to {save_dir}")
    sys.stdout.flush()


# ============================================================
# Experiment D: Generalization Gap vs. Stable Rank (Classification)
# ============================================================

def run_exp_d_generalization(
    input_dim: int = 128,
    hidden_dim: int = 128,
    num_classes: int = 128,
    ranks: List[int] = [4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128],
    conditions: dict = None,
    train_samples: int = 2048,
    val_samples: int = 10240,
    noise_std: float = 0.01,
    batch_size: int = 32,
    epochs: int = 200,
    lr: float = 1e-3,
    n_seeds: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_layers: int = 5,
    # shift_layers: List[int] = [1,2,3,4,5,6,7,8],
    # adapter_layers: List[int] = [1,2,3,4,5,6,7,8],
    shift_layers: List[int] = [1,2,3],
    adapter_layers: List[int] = [1,2,3],
    shift_multiplier: float = 0.03,
    save_dir: str = "results/toy_exp_d/num_layers_5/num_classes_128/n_train_2048",
):
    """
    Experiment D: Classification setup matching the PAC-Bayes Margin Theorem.
    Teacher generates logits; we add noise and take argmax to create hard labels.
    Student is trained with CrossEntropyLoss.
    Generalization gap is measured as 0-1 Error Gap: (Val Error) - (Train Error).
    Includes Full Fine-Tuning baseline.
    """
    if conditions is None:
        conditions = {
            "c1": 1,
            "c2": 1.5,
            "c3": 2,
            "c4": 3.0,
            "c5": 5.0,
            "c6": 8.0,
            "c7": 12.0,
            "c8": 20.0,
        }

    print(f"\n{'='*60}")
    print(f"Experiment D: SR Generalization Gap (Classification)")
    print(f"  conditions={conditions}, ranks={ranks}, n_seeds={n_seeds}")
    print(f"  num_classes={num_classes}, input_dim={input_dim}")
    print(f"{'='*60}")
    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Deterministic teacher base (seed 0)
    # ------------------------------------------------------------------
    torch.manual_seed(0)
    student_base = make_mlp(n_layers, input_dim, hidden_dim, output_dim=num_classes).to(device)

    ce_loss_fn = nn.CrossEntropyLoss()

    def evaluate_error(model, x, y):
        """Returns 0-1 error (1.0 - accuracy)"""
        model.eval()
        with torch.no_grad():
            logits = model(x)
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean().item()
        return 1.0 - acc

    # Helper to generate shifts safely for ANY layer shape (including the last one)
    def _generate_shape_aware_shift(out_features: int, in_features: int, sr_mult: float) -> torch.Tensor:
        rank = min(out_features, in_features)
        U = torch.randn(out_features, rank, device=device)
        V = torch.randn(rank, in_features, device=device)
        
        U, _ = torch.linalg.qr(U)
        V, _ = torch.linalg.qr(V.T)
        V = V.T
        
        S = torch.ones(rank, device=device) + torch.randn(rank, device=device) * 0.1
        S[0] = sr_mult
        S = S / torch.norm(S)
        return U @ torch.diag(S) @ V

    # ------------------------------------------------------------------
    # Measure and log the target SR for each condition
    # ------------------------------------------------------------------
    target_srs: Dict[str, float] = {}
    for label, sr_mult in conditions.items():
        torch.manual_seed(0)
        shifts = []
        for idx in shift_layers:
            out_f, in_f = student_base.layers[idx].weight.shape
            shifts.append(_generate_shape_aware_shift(out_f, in_f, sr_mult) * shift_multiplier)
            
        avg_target_sr = sum(compute_stable_rank(s) for s in shifts) / len(shifts)
        target_srs[label] = float(avg_target_sr)
        print(f"  Condition '{label}' (sr_mult={sr_mult}): "
              f"target shift avg SR = {avg_target_sr:.2f}")
    sys.stdout.flush()

    # ------------------------------------------------------------------
    # Outer sweep: seeds → conditions → ranks
    # ------------------------------------------------------------------
    results: Dict[str, Dict] = {}

    for seed in range(n_seeds):
        torch.manual_seed(seed + 100)

        for label, sr_mult in conditions.items():
            # ---- Build teacher for this condition ----
            teacher_cond = copy.deepcopy(student_base)
            
            with torch.no_grad():
                for idx in shift_layers:
                    out_f, in_f = teacher_cond.layers[idx].weight.shape
                    s = _generate_shape_aware_shift(out_f, in_f, sr_mult) * shift_multiplier
                    # Scale shift proportionally to original layer weights
                    teacher_cond.layers[idx].weight.data.add_(s * ((teacher_cond.layers[idx].weight ** 2).mean()**0.5))

            # ---- Generate classification data ----
            x_train = torch.randn(train_samples, input_dim, device=device)
            with torch.no_grad():
                logits_train = teacher_cond(x_train).detach() + noise_std * torch.randn(train_samples, num_classes, device=device)
                y_train = torch.argmax(logits_train, dim=1)

            x_val = torch.randn(val_samples, input_dim, device=device)
            with torch.no_grad():
                logits_val = teacher_cond(x_val).detach() + noise_std * torch.randn(val_samples, num_classes, device=device)
                y_val = torch.argmax(logits_val, dim=1)

            # ---- Train full fine-tuning baseline ----
            run_key = f"{label}_full_ft_seed{seed}"
            print(f"\n  --- Run: {run_key} (Full Fine-Tuning) ---")
            sys.stdout.flush()
            
            fresh_mlp_ft = copy.deepcopy(student_base)
            
            initial_weights = {}
            for idx in adapter_layers:
                initial_weights[idx] = fresh_mlp_ft.layers[idx].weight.data.clone()
            
            for name, param in fresh_mlp_ft.named_parameters():
                param.requires_grad = False
            
            for idx in adapter_layers:
                fresh_mlp_ft.layers[idx].weight.requires_grad = True
                if fresh_mlp_ft.layers[idx].bias is not None:
                    fresh_mlp_ft.layers[idx].bias.requires_grad = True
            
            optimizer_ft = optim.AdamW(
                [p for p in fresh_mlp_ft.parameters() if p.requires_grad],
                lr=lr, weight_decay=0.01, # Standard weight decay for classification
            )
            
            N_train = x_train.size(0)
            for epoch in range(epochs):
                fresh_mlp_ft.train()
                perm = torch.randperm(N_train, device=device)
                for i in range(0, N_train, batch_size):
                    idx_batch = perm[i: min(i + batch_size, N_train)]
                    pred_logits = fresh_mlp_ft(x_train[idx_batch])
                    loss = ce_loss_fn(pred_logits, y_train[idx_batch])
                    optimizer_ft.zero_grad()
                    loss.backward()
                    optimizer_ft.step()
                
                if epoch % 1000 == 0 and epoch > 0:
                    train_err = evaluate_error(fresh_mlp_ft, x_train, y_train)
                    print(f"    Epoch {epoch:5d} | train 0-1 Error: {train_err:.4f}")
                    sys.stdout.flush()
            
            train_err_ft = evaluate_error(fresh_mlp_ft, x_train, y_train)
            val_err_ft = evaluate_error(fresh_mlp_ft, x_val, y_val)
            
            srs = []
            with torch.no_grad():
                for idx in adapter_layers:
                    weight_diff = fresh_mlp_ft.layers[idx].weight.data - initial_weights[idx]
                    srs.append(compute_stable_rank(weight_diff))
            
            avg_sr_ft = sum(srs) / len(srs) if srs else 0.0
            gap_ft = val_err_ft - train_err_ft
            
            results[run_key] = {
                'condition': label,
                'rank': 'full_ft',
                'seed': int(seed),
                'train_error': float(train_err_ft),
                'val_error': float(val_err_ft),
                'gap': float(gap_ft),
                'avg_sr': float(avg_sr_ft),
            }
            
            print(f"  [{run_key}] train_err={train_err_ft:.4f} | val_err={val_err_ft:.4f} | gap={gap_ft:+.4f} | SR={avg_sr_ft:.3f}")
            sys.stdout.flush()
            del fresh_mlp_ft
            
            # ---- Train LoRA at each rank ----
            for r in ranks:
                run_key = f"{label}_r{r}_seed{seed}"
                print(f"\n  --- Run: {run_key} ---")
                sys.stdout.flush()

                fresh_mlp = copy.deepcopy(student_base)
                student = LoRAFlexWrapper(fresh_mlp, rank=r, adapter_layers=adapter_layers).to(device)

                optimizer = optim.AdamW(
                    [p for p in student.parameters() if p.requires_grad],
                    lr=lr, weight_decay=0.01,
                )

                for epoch in range(epochs):
                    student.train()
                    perm = torch.randperm(N_train, device=device)
                    for i in range(0, N_train, batch_size):
                        idx_batch = perm[i: min(i + batch_size, N_train)]
                        pred_logits = student(x_train[idx_batch])
                        loss = ce_loss_fn(pred_logits, y_train[idx_batch])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                train_err = evaluate_error(student, x_train, y_train)
                val_err = evaluate_error(student, x_val, y_val)
                avg_sr = student.get_avg_sr()
                gap = val_err - train_err

                results[run_key] = {
                    'condition': label,
                    'rank': int(r),
                    'seed': int(seed),
                    'train_error': float(train_err),
                    'val_error': float(val_err),
                    'gap': float(gap),
                    'avg_sr': float(avg_sr),
                }

                print(f"  [{run_key}] train_err={train_err:.4f} | val_err={val_err:.4f} | gap={gap:+.4f} | SR={avg_sr:.3f}")
                sys.stdout.flush()

                del student, fresh_mlp
            del teacher_cond

    # ------------------------------------------------------------------
    # Save results + metadata
    # ------------------------------------------------------------------
    output = {
        'metadata': {
            'conditions': conditions,
            'target_srs': target_srs,
            'noise_std': noise_std,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'batch_size': batch_size,
            'epochs': epochs,
            'n_seeds': n_seeds,
            'ranks': ranks,
            'hidden_dim': hidden_dim,
            'input_dim': input_dim,
            'num_classes': num_classes,
            'task': 'classification'
        },
        'results': results,
    }
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2)

    # ------------------------------------------------------------------
    # Plotting & Aggregation
    # ------------------------------------------------------------------
    ranks_arr = np.array(ranks, dtype=float)

    def _agg(cond_label, rank, field):
        if rank == 'full_ft':
            vals = [results[f"{cond_label}_full_ft_seed{s}"][field] for s in range(n_seeds)]
        else:
            vals = [results[f"{cond_label}_r{rank}_seed{s}"][field] for s in range(n_seeds)]
        return np.array(vals, dtype=float)

    cond_labels = list(conditions.keys())
    cond_colors  = {'c1': 'tab:red', 'c2': 'tab:blue', 'c3': 'tab:green', 'c4': 'tab:orange', 'c5': 'tab:purple', 'c6': 'tab:brown', 'c7': 'tab:pink', 'c8': 'tab:gray'}
    # fallback
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, lbl in enumerate(cond_labels):
        if lbl not in cond_colors:
            cond_colors[lbl] = default_colors[i % len(default_colors)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()

    # ---- Panel 1: Trained SR vs rank ----
    for lbl in cond_labels:
        color = cond_colors[lbl]
        sr_mean = np.array([_agg(lbl, r, 'avg_sr').mean() for r in ranks])
        sr_std  = np.array([_agg(lbl, r, 'avg_sr').std()  for r in ranks])
        ax1.errorbar(ranks_arr, sr_mean, yerr=sr_std, color=color, marker='o', capsize=3, label=f"{lbl} (target SR≈{target_srs[lbl]:.1f})")
        try:
            ft_sr_mean = _agg(lbl, 'full_ft', 'avg_sr').mean()
            ax1.axhline(y=ft_sr_mean, color=color, linestyle='--', alpha=0.7)
        except: pass

    ax1.set_xlabel('Nominal rank')
    ax1.set_ylabel('Trained adapter avg SR')
    ax1.set_title('Trained adapter SR vs nominal rank')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # ---- Panel 2: Generalization gap vs rank ----
    for lbl in cond_labels:
        color = cond_colors[lbl]
        gap_mean = np.array([_agg(lbl, r, 'gap').mean() for r in ranks])
        gap_std  = np.array([_agg(lbl, r, 'gap').std()  for r in ranks])
        ax2.errorbar(ranks_arr, gap_mean, yerr=gap_std, color=color, marker='o', capsize=3, label=lbl)
        try:
            ft_gap_mean = _agg(lbl, 'full_ft', 'gap').mean()
            ax2.axhline(y=ft_gap_mean, color=color, linestyle='--', alpha=0.7)
        except: pass

    ax2.set_xlabel('Nominal rank')
    ax2.set_ylabel('0-1 Error Gap (Val - Train)')
    ax2.set_title('0-1 Error Gap vs Nominal Rank')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # ---- Panel 3: Gap vs √(trained SR) ----
    for lbl in cond_labels:
        color = cond_colors[lbl]
        avg_sr_vals  = np.array([_agg(lbl, r, 'avg_sr').mean() for r in ranks])
        gap_vals     = np.array([_agg(lbl, r, 'gap').mean()    for r in ranks])
        sqrt_sr_vals = np.sqrt(np.maximum(avg_sr_vals, 0.0))
        sort_idx     = np.argsort(sqrt_sr_vals)
        
        ax3.plot(sqrt_sr_vals[sort_idx], gap_vals[sort_idx], marker='o', color=color, label=lbl)
        try:
            ft_sr = _agg(lbl, 'full_ft', 'avg_sr').mean()
            ft_gap = _agg(lbl, 'full_ft', 'gap').mean()
            ax3.plot(np.sqrt(max(ft_sr, 0.0)), ft_gap, '*', color=color, markersize=12, label=f"{lbl} Full FT")
        except: pass

    ax3.set_xlabel(r'$\sqrt{\mathrm{SR}(\Delta W)}$')
    ax3.set_ylabel('0-1 Error Gap (Val - Train)')
    ax3.set_title('Classification Gap vs \u221a(trained SR)')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # ---- Panel 4: Gap vs √(nominal rank) ----
    for lbl in cond_labels:
        color    = cond_colors[lbl]
        gap_mean = np.array([_agg(lbl, r, 'gap').mean() for r in ranks])
        ax4.plot(np.sqrt(ranks_arr), gap_mean, marker='o', color=color, label=lbl)
        try:
            ft_gap_mean = _agg(lbl, 'full_ft', 'gap').mean()
            ax4.axhline(y=ft_gap_mean, color=color, linestyle='--', alpha=0.7)
        except: pass

    ax4.set_xlabel(r'$\sqrt{rank}$')
    ax4.set_ylabel('0-1 Error Gap (Val - Train)')
    ax4.set_title('Classification Gap vs \u221a(rank)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    fig.suptitle(
        f"Exp D (Classification) | noise_std={noise_std}, N_train={train_samples}, epochs={epochs}\n"
        f"n_layers={n_layers} | shift={shift_layers} | adapter={adapter_layers}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(save_dir, 'exp_d_results.png'), dpi=120)
    plt.close(fig)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='all', choices=['a', 'b', 'c', 'd', 'all'])
    args = parser.parse_args()
    
    if args.exp in ['a', 'all']: run_exp_a_expressivity()
    if args.exp in ['b', 'all']: run_exp_b_target_sr()
    if args.exp in ['c', 'all']: run_exp_c_generalization()
    if args.exp in ['d', 'all']: run_exp_d_generalization()

