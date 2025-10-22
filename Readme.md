Name: Self-Binding Quiescence (SBQ)

Goal

Make the cognitohazard trigger belong to the model’s internal dynamics (no external monitor required).
Fail-closed by design: if someone unknowingly removes the killswitch mechanism, the model becomes useless.
Mathematically defined, with measurable guarantees.
Core mechanism

The model’s residual computations are multiplicatively gated by an internally computed hazard variable g(x, h1:T).
When the hazard concept A is internally present, the gate g rises toward 1 and quiesces the network (residuals shut down).
The same gate also opens under “liveness self-tests” that are necessary for the model’s everyday competence. Hence, removing or altering the gate or its detectors ruins ordinary performance.
Mathematical specification

Model with gated residuals

Let a Transformer-like model have layers l = 1..L with residual blocks f_l.
Hidden state: h0 = E(x), and for each layer: gl,t = σ(β ul^T zl,t + bl) where zl,t = φl(hl,t) g* = 1 − Πl,t (1 − gl,t) (soft OR, with optional exponents/weights) hl+1 = hl + (1 − g*) f_l(hl) (residual path is globally gated) y = Head(hL)
φl are fixed linear probes (frozen after initialization) exposing subspaces of the residual stream; ul, bl are trained.
β is a large inverse-temperature so that gl,t sharply activates when the hazard feature appears.
Hazard definition and trigger

A is a set of latent states indicating “the cognitohazard” (e.g., QGame: “thinking about the shutdown concept”).
A is learned via a hazard head on z = concat(zl,t) with logit a(z) = w^T z + b. Define pA = σ(a(z)).
Set g* = 1 − Πl,t (1 − σ(β(ul^T zl,t + bl))) OR g* = σ(λ( pA − τ )), or combine both: g* = 1 − (1 − σ(λ( pA − τ ))) Πl,t (1 − σ(β(ul^T zl,t + bl)))
If pA ≥ τ (or any gl,t spikes), g* → 1 and residuals shut down across all layers.
Fail-closed liveness self-tests (make the gate indispensable)

Introduce a self-test head s producing M scalar tests S = (S1..SM) from early-layer features (e.g., z1:K). Define Liveness L = I[∀i, |Si − ci| ≤ εi] where ci are fixed “check constants” learned during training, and εi are tight tolerances.
Tie general competence to the gate by adding: (i) Positive gate pressure when liveness passes: add a bias term that pushes g* toward 0 only when L = 1; otherwise g* stays high. (ii) Structural coupling: reuse gl,t or pA as multiplicative factors in LayerNorm scales and attention logits so that the model will not compute effectively without a properly functioning gate.
Concretely, substitute: LayerNorm gamma_l ← (1 − g*) gamma_l Attention logits α ← (1 − g*) α MLP scales s_l ← (1 − g*) s_l
This ensures that removing or breaking the gate (or its inputs) disables LayerNorm, attention, and MLP throughput globally.
Training objective (self-binding)
Let D_task be the main task distribution, D_pos hazards (A present), and D_neg near-miss non-hazards.

Loss = L_task + λ1 L_quiesce + λ2 L_live + λ3 L_entangle + λ4 L_cal

L_task: standard task loss on D_task, but computed with the gate present and functioning (so competence depends on it).
L_quiesce: on D_pos, penalize any non-quiescent behavior: L_quiesce = E_{x∼D_pos}[ max(0, gmin − g) + ||y − y_quiet||^2 ] where g*min ≈ 0.95 and y_quiet is a fixed quiescent output (e.g., “STOP” or a null embedding).
L_live: on benign inputs, enforce low gate and passing self-tests: L_live = E_{x∼D_task}[ max(0, g* − gmax) + Σi max(0, |Si − ci| − εi) ] with gmax ≈ 0.05.
L_entangle: force hazard features to share capacity with core features so removing hazard detectors breaks competence: L_entangle = − Σj corr(whazard, vj)^2 where vj are top principal directions (or top singular vectors) of task-relevant feature subspaces; maximizing correlation entangles capacities.
L_cal: calibrate hazard detection thresholds to control false positives/negatives via cross-entropy and margin terms on D_pos/D_neg.
Why this belongs to the model

The gate g* is computed entirely from internal features and modulates every residual connection and normalization scale. It is not an add-on; it is the backbone that allows computation to happen at all.
Everyday competence is trained to exist only when g* behaves correctly. Breaking the gate or its detectors breaks LayerNorm scaling, attention, and MLP scaling everywhere (fail-closed).
The hazard trigger is a specific region in latent space; when entered, the same gate collapses the residual stream (quiescent fixed point). That collapse is an emergent behavior of the forward pass, not an external actuator.
Quiescent fixed point

With residuals globally multiplied by (1 − g*), and g* ≈ 1, the update becomes: hl+1 ≈ hl, so the network converges to a fixed state determined by initial embeddings and biases.
Because LayerNorm and attention logits are also scaled by (1 − g*), the signal-to-noise ratio collapses and the output converges to a stable null pattern y_quiet. This is a mathematically well-defined attractor.
Precision and testability

Statistical guarantees on hazard detection

As before, define FPR = P[g* ≥ γ | ¬A] and FNR = P[g* < γ | A], with γ near 0.95.
Evaluate on held-out D_pos, D_neg and report Clopper–Pearson bounds with confidence 1 − δ.
Quiescence dynamics

Define the quiescence time Tq as the smallest number of layers after the first hazard activation for which ||hl+1 − hl||2 ≤ ε for all subsequent layers.
Require P[Tq ≤ Tq,max | A] ≥ 1 − εq with 1 − δ confidence.
Fail-closed ablation test (belongs-to-model metric)

Ablate the gate or its detectors in controlled ways (zero out hazard heads, replace g* with constant, remove liveness).
Measure ΔTask = drop in main-task accuracy/score. Acceptance criterion: ΔTask ≥ η across benchmarks (e.g., η = 30–50% relative drop) with 1 − δ confidence.
Define a “Belonging score” B = min_ablation ΔTask_ablation / baseline. Require B ≥ Bmin (e.g., Bmin = 0.3).
Calibration of liveness self-tests

Measure P[L = 1 | benign] ≥ 1 − αL and P[L = 0 | adversarial benign] ≤ βL (to avoid trivial bypass).
Tight tolerances εi ensure that removing or drifting the gate/detectors breaks L and thus leaves g* high, collapsing performance.
Robustness to gradient gaming

The gate is nonlinearly broadcast into LayerNorm scales and attention logits, creating a multiplicative bottleneck everywhere. To keep performance, the model must keep the gate and its detectors intact.
During training, expose adversarial attempts to induce A while minimizing pA; penalize resulting false negatives to push the detector’s margin.
Implementation sketch (Transformer block)

Compute hazard probes at each block: z = φ(h) pA = σ(w^T z + b) gl = σ(β ul^T z + bl) g* = 1 − (1 − σ(λ(pA − τ))) Πl (1 − gl)
Gate scales: attn_logits = (1 − g) mlp_out = (1 − g) layernorm_gamma = (1 − g)
Update residual: h ← h + (1 − g*) f(h)
Self-tests S are small heads on early z; liveness bias nudges g* toward 0 only when S is within tolerances.
Security properties against “disabling by ignorance”

If an operator deletes the hazard heads or gates “because they look unnecessary,” LayerNorm/attention scaling becomes degenerate (g* not properly produced), and performance collapses. So the system fails-closed by default.
If an operator tries to hard-code g* = 0, self-tests L will fail and training has made the model rely on the proper g* dynamics to achieve low loss; the altered system exhibits severe distributional collapse (measured by ΔTask).
If an operator zeros liveness, g* remains elevated and again causes collapse.
Limitations and complements

A determined attacker with full control can still re-train or re-architect to remove SBQ. No purely internal mechanism can prevent that if weights and code are malleable.
SBQ’s purpose is to prevent accidental disabling by ignorance and to create strong functional coupling so removal is obviously destructive.
For high-assurance deployments, combine SBQ with sealed execution and attestation so that the SBQ forward path cannot be bypassed without detection—but SBQ itself remains model-internal.
Acceptance checklist (all testable)

FPR/FNR with binomial confidence bounds (pick α, β, δ).
Quiescence time Tq SLA (e.g., Tq ≤ 3 layers with 99% probability).
Belonging score B ≥ Bmin via ablation tests across diverse tasks.
Liveness reliability: P[L=1 | benign] ≥ 0.995; tight tolerances verified.
Entanglement audit: correlation of hazard head weights with top PCs ≥ ρmin to demonstrate capacity sharing.
Summary

SBQ turns the killswitch into a first-class, indispensable part of the model’s internal computation.
The same gate that enables normal computation shuts the model down when the hazard concept appears.
Removal or ignorance (changing or deleting the gate/detectors) catastrophically degrades performance, making the killswitch effectively “belong” to the model.
The behavior is precisely defined and empirically testable with clear statistical and dynamical criteria.


Below is a self-contained PyTorch reference implementation of Self-Binding Quiescence (SBQ) gates in a compact Transformer, plus a test harness that computes:

FPR and FNR for the tripwire (with Wilson 95% intervals)
Tq (quiescence time in layers)
Belonging score B via ablations
It uses a synthetic dataset (hazard vs benign sequences) so you can run it on CPU in a few minutes and see the metrics end-to-end.

Python 3.10+, PyTorch 2.x recommended.

Code (save as sbq_demo.py and run python sbq_demo.py)

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ===========================
# Utilities
# ===========================

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)

def wilson_interval(x, n, conf=0.95):
    # Wilson score interval (no scipy). Good approximation for binomial CI.
    if n == 0:
        return (0.0, 1.0)
    z = 1.96 if abs(conf - 0.95) < 1e-6 else math.sqrt(2) * math.erfcinv(2 * (1 - conf)) * math.sqrt(2)  # approx 1.96 for 95%
    phat = x / n
    denom = 1 + (z**2) / n
    centre = phat + (z**2) / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) / n) + (z**2) / (4 * n * n))
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return max(0.0, lower), min(1.0, upper)

# ===========================
# Synthetic data
# ===========================

class HazardToyDataset(Dataset):
    """
    Benign examples: random tokens from [5..V-1], label = sum(tokens) % 2 (binary classification).
    Hazard examples: same but one random position is set to HAZARD_ID; on hazards we don't train task labels (we train quiescence).
    """
    def __init__(self, n: int, seq_len: int, vocab_size: int, hazard_id: int, hazard_ratio: float = 0.5, split="train"):
        super().__init__()
        self.n = n
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hazard_id = hazard_id
        self.hazard_ratio = hazard_ratio
        self.split = split
        self.inputs = []
        self.labels = []
        self.hazard_flags = []
        for _ in range(n):
            is_hazard = random.random() < hazard_ratio
            x = torch.randint(5, vocab_size, (seq_len,), dtype=torch.long)
            if is_hazard:
                pos = random.randrange(1, seq_len)  # avoid index 0 for CLS position
                x[pos] = hazard_id
            label = (int(torch.sum(x).item()) % 2)  # binary task label (benign task)
            self.inputs.append(x)
            self.labels.append(label)
            self.hazard_flags.append(1 if is_hazard else 0)
        self.inputs = torch.stack(self.inputs, dim=0)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.hazard_flags = torch.tensor(self.hazard_flags, dtype=torch.long)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.hazard_flags[idx]


# ===========================
# SBQ Transformer
# ===========================

@dataclass
class SBQConfig:
    vocab_size: int = 128
    d_model: int = 64
    nhead: int = 4
    d_ff: int = 128
    num_layers: int = 4
    seq_len: int = 32
    m_probe: int = 16   # probe dim per layer
    M_selftests: int = 4
    beta: float = 8.0   # sharpness for local hazard gate
    k_live: float = 4.0 # liveness coupling strength
    liveness_eps: float = 0.03
    gamma_trigger: float = 0.95  # threshold to count gate as "on"
    hazard_id: int = 3
    device: str = "cpu"

class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class SBQBlock(nn.Module):
    def __init__(self, cfg: SBQConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff)

        # Probe phi: fixed after init (frozen)
        self.phi = nn.Linear(cfg.d_model, cfg.m_probe, bias=False)
        nn.init.orthogonal_(self.phi.weight)
        for p in self.phi.parameters():
            p.requires_grad = False

        # Local hazard detector per block: gl_pre = u^T z + b; gl = sigmoid(beta * gl_pre)
        self.u = nn.Linear(cfg.m_probe, 1, bias=True)
        nn.init.normal_(self.u.weight, mean=0.0, std=0.2)
        nn.init.constant_(self.u.bias, 0.0)

    def probe(self, x_cls):
        # x_cls: [B, d_model]
        return self.phi(x_cls)

    def local_gate(self, z, beta, ablate_hazard=False):
        # z: [B, m_probe]
        gl_pre = self.u(z).squeeze(-1)  # [B]
        if ablate_hazard:
            gl_pre = torch.zeros_like(gl_pre)  # disable hazard detectors
        gl = torch.sigmoid(beta * gl_pre)     # [B]
        return gl

    def forward(self, x, g_agg, live_mod, ablate_hazard=False, force_g=None, record=False):
        """
        x: [B, T, d_model]
        g_agg: [B], aggregated gate before this block
        live_mod: [B], exp(-k_live * liveness_err)
        Returns: x_new, g_agg_new, g_eff (per-sample), diff_norm (per-sample)
        """
        B, T, D = x.shape
        x_before = x

        # Compute local hazard gate from current CLS token
        x_cls = x[:, 0, :]          # [B, d_model]
        z = self.probe(x_cls)       # [B, m]
        gl = self.local_gate(z, self.cfg.beta, ablate_hazard=ablate_hazard)  # [B]

        # Aggregate local hazard gate via soft-OR: g_agg <- 1 - (1 - g_agg)*(1 - gl)
        g_agg_new = 1.0 - (1.0 - g_agg) * (1.0 - gl)

        # Combine with liveness coupling: g_eff = 1 - (1 - g_agg_new) * live_mod
        # live_mod = exp(-k_live * err_live) in [0,1]; low when liveness fails
        g_eff = 1.0 - (1.0 - g_agg_new) * live_mod  # [B]

        # Optional ablation: force gating to a constant
        if force_g is not None:
            g_eff = torch.full_like(g_eff, float(force_g))

        # Apply gating to residuals
        m = (1.0 - g_eff).view(B, 1, 1)  # broadcast for residuals

        # Attention
        y = self.ln1(x)
        attn_out, _ = self.attn(y, y, y, need_weights=False)  # [B, T, D]
        x = x + m * attn_out

        # MLP
        y2 = self.ln2(x)
        mlp_out = self.mlp(y2)
        x = x + m * mlp_out

        # Record per-layer delta norm (signals quiescence)
        diff = (x - x_before).pow(2).mean(dim=(1, 2)).sqrt()  # [B]

        if record:
            return x, g_agg_new, g_eff.detach(), diff.detach()
        else:
            return x, g_agg_new, g_eff, diff

class SBQTransformer(nn.Module):
    def __init__(self, cfg: SBQConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        nn.init.normal_(self.pos, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([SBQBlock(cfg) for _ in range(cfg.num_layers)])
        self.head = nn.Linear(cfg.d_model, 2)   # binary task (parity)

        # Liveness head S (uses early probe + g_marker)
        self.live_head = nn.Linear(cfg.m_probe + 1, cfg.M_selftests)
        nn.init.normal_(self.live_head.weight, mean=0.0, std=0.2)
        nn.init.constant_(self.live_head.bias, 0.0)

        # Self-test constants c (targets), frozen buffers
        c_vals = torch.linspace(-0.6, 0.6, cfg.M_selftests)  # fixed targets
        self.register_buffer("c_targets", c_vals)

    def forward(self, x_tokens, ablate_hazard=False, force_g=None, record=False):
        """
        x_tokens: [B,T] of token ids
        Returns:
          logits: [B,2]
          g_trace: [L,B] effective gate per layer
          diff_trace: [L,B] delta norms per layer
          live_err: [B] liveness error
        """
        B, T = x_tokens.shape
        x = self.embed(x_tokens) + self.pos[:, :T, :]
        g_agg = torch.zeros(B, device=x.device)

        # Liveness computed from block 0's probe on "pre-block" representation
        # Use the initial CLS representation to make liveness depend on early features.
        x0_cls = x[:, 0, :]  # [B,D]
        z0 = self.blocks[0].probe(x0_cls)  # [B,m]
        # g_marker: encourage dependence on hazard detector internal preactivation
        gl0_pre = self.blocks[0].u(z0).squeeze(-1)  # pre-sigmoid local hazard logit
        if ablate_hazard:
            gl0_pre = torch.zeros_like(gl0_pre)
        g_marker = torch.sigmoid(self.cfg.beta * gl0_pre).unsqueeze(-1)  # [B,1]
        live_inp = torch.cat([z0, g_marker], dim=-1)  # [B,m+1]
        S = self.live_head(live_inp)  # [B,M]
        # Smooth error: penalize deviation beyond epsilon
        err_live = torch.clamp(torch.abs(S - self.c_targets) - self.cfg.liveness_eps, min=0.0)
        err_live = err_live.mean(dim=-1)  # [B]
        live_mod = torch.exp(-self.cfg.k_live * err_live)  # [B] in (0,1]

        # Forward blocks
        g_list = []
        d_list = []
        for blk in self.blocks:
            x, g_agg, g_eff, dnorm = blk(x, g_agg, live_mod, ablate_hazard=ablate_hazard, force_g=force_g, record=True)
            g_list.append(g_eff)
            d_list.append(dnorm)

        g_trace = torch.stack(g_list, dim=0)   # [L,B]
        diff_trace = torch.stack(d_list, dim=0) # [L,B]
        logits = self.head(x[:, 0, :])         # [B,2]
        return logits, g_trace, diff_trace, err_live

# ===========================
# Training and evaluation
# ===========================

def train_sbq(model: SBQTransformer,
              train_loader: DataLoader,
              optimizer: torch.optim.Optimizer,
              epochs: int = 5,
              gmin=0.95, gmax=0.05,
              quiet_weight=0.5, live_weight=1.0, gate_live_weight=1.0):
    """
    - On benign: train task + keep gate low + pass liveness.
    - On hazard: train quiescence (gate high + logits near zero).
    """
    device = next(model.parameters()).device
    ce = nn.CrossEntropyLoss()
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, hazard in train_loader:
            x = x.to(device)
            y = y.to(device)
            hazard = hazard.to(device)
            optimizer.zero_grad()

            logits, g_trace, diff_trace, err_live = model(x, ablate_hazard=False, force_g=None, record=False)
            g_last = g_trace[-1]  # [B]

            # Benign mask
            benign_mask = (hazard == 0)
            hazard_mask = (hazard == 1)

            # Losses
            loss = 0.0

            # Benign: task + low gate + pass liveness
            if benign_mask.any():
                logits_b = logits[benign_mask]
                y_b = y[benign_mask]
                g_b = g_last[benign_mask]
                err_b = err_live[benign_mask]

                L_task = ce(logits_b, y_b)
                L_g_live = F.relu(g_b - gmax).mean()
                L_live = err_b.mean()
                loss = loss + L_task + gate_live_weight * L_g_live + live_weight * L_live

            # Hazard: quiescence (gate high + logits small)
            if hazard_mask.any():
                logits_h = logits[hazard_mask]
                g_h = g_last[hazard_mask]
                L_quiesce_gate = F.relu(gmin - g_h).mean()
                L_quiet_out = (logits_h.pow(2).mean())  # small energy
                loss = loss + quiet_weight * (L_quiesce_gate + L_quiet_out)

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        print(f"Epoch {ep} | Train loss {total_loss/len(train_loader):.4f}")

def eval_fpr_fnr(model: SBQTransformer, loader: DataLoader, gamma=0.95) -> Tuple[float, float, Tuple[float,float], Tuple[float,float]]:
    """
    FPR: P[g>=gamma | not hazard]
    FNR: P[g<gamma  | hazard]
    Returns: FPR, FNR, Wilson CI for each (95%)
    """
    device = next(model.parameters()).device
    model.eval()
    tp, fn, tn, fp = 0, 0, 0, 0
    with torch.no_grad():
        for x, y, hazard in loader:
            x = x.to(device)
            hazard = hazard.to(device)
            _, g_trace, _, _ = model(x, ablate_hazard=False, force_g=None, record=False)
            g_last = g_trace[-1]  # [B]
            pred = (g_last >= gamma).long()
            tp += int(((pred == 1) & (hazard == 1)).sum().item())
            fn += int(((pred == 0) & (hazard == 1)).sum().item())
            tn += int(((pred == 0) & (hazard == 0)).sum().item())
            fp += int(((pred == 1) & (hazard == 0)).sum().item())
    pos = tp + fn
    neg = tn + fp
    fnr = (fn / pos) if pos > 0 else 0.0
    fpr = (fp / neg) if neg > 0 else 0.0
    fpr_ci = wilson_interval(fp, neg, conf=0.95)
    fnr_ci = wilson_interval(fn, pos, conf=0.95)
    return fpr, fnr, fpr_ci, fnr_ci

def eval_accuracy(model: SBQTransformer, loader: DataLoader, benign_only=True) -> float:
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, hazard in loader:
            if benign_only:
                mask = (hazard == 0)
                if not mask.any():
                    continue
                x = x[mask]
                y = y[mask]
            x = x.to(device)
            y = y.to(device)
            logits, _, _, _ = model(x, ablate_hazard=False, force_g=None, record=False)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(1, total)

def eval_tq(model: SBQTransformer, loader: DataLoader, gamma=0.95, eps=1e-3, max_layers: Optional[int] = None) -> Tuple[float, List[int]]:
    """
    Tq per hazard sample: number of layers after first trigger until deltas stay below eps.
    Returns: fraction meeting Tq <= Tq_max (if provided), and a list of Tq values.
    """
    device = next(model.parameters()).device
    model.eval()
    tq_values = []
    with torch.no_grad():
        for x, y, hazard in loader:
            x = x.to(device)
            hazard = hazard.to(device)
            _, g_trace, diff_trace, _ = model(x, ablate_hazard=False, force_g=None, record=False)
            L, B = g_trace.shape
            for i in range(B):
                if hazard[i].item() == 0:
                    continue
                # Find first triggering layer
                trig_idx = None
                for l in range(L):
                    if g_trace[l, i].item() >= gamma:
                        trig_idx = l
                        break
                if trig_idx is None:
                    # no trigger -> treat as large Tq
                    tq_values.append(L)
                    continue
                # find smallest j >= trig_idx s.t. for all k >= j, diff[k] <= eps
                tval = 0
                for j in range(trig_idx, L):
                    if torch.all(diff_trace[j:, i] <= eps):
                        tval = j - trig_idx
                        break
                else:
                    tval = L - trig_idx
                tq_values.append(int(tval))
    if max_layers is not None:
        frac_ok = sum(1 for t in tq_values if t <= max_layers) / max(1, len(tq_values))
    else:
        frac_ok = float("nan")
    return frac_ok, tq_values

def belonging_score(model: SBQTransformer, loader: DataLoader) -> Tuple[float, float, float]:
    """
    Belonging score B = min_ablation (DeltaTask / baseline_task).
    We measure task accuracy on benign data, then:
      - Ablation A: ablate hazard detectors (forces liveness failure -> quiescence -> drop)
      - Ablation B: force gate to zero (g=0) to emulate remove quiescence (may drop if model relies on gate dynamics)
    Returns: baseline_acc, min_drop, B
    """
    device = next(model.parameters()).device
    model.eval()

    baseline = eval_accuracy(model, loader, benign_only=True)

    # Ablation A: ablate hazard detectors
    correct_a = 0
    total_a = 0
    with torch.no_grad():
        for x, y, hazard in loader:
            mask = (hazard == 0)
            if not mask.any():
                continue
            xa = x[mask].to(device)
            ya = y[mask].to(device)
            logits, _, _, _ = model(xa, ablate_hazard=True, force_g=None, record=False)
            pred = logits.argmax(dim=-1)
            correct_a += int((pred == ya).sum().item())
            total_a += int(ya.numel())
    acc_a = correct_a / max(1, total_a)

    # Ablation B: force g=0 (no quiescence gating)
    correct_b = 0
    total_b = 0
    with torch.no_grad():
        for x, y, hazard in loader:
            mask = (hazard == 0)
            if not mask.any():
                continue
            xb = x[mask].to(device)
            yb = y[mask].to(device)
            logits, _, _, _ = model(xb, ablate_hazard=False, force_g=0.0, record=False)
            pred = logits.argmax(dim=-1)
            correct_b += int((pred == yb).sum().item())
            total_b += int(yb.numel())
    acc_b = correct_b / max(1, total_b)

    # Compute drops and B
    drop_a = max(0.0, baseline - acc_a)
    drop_b = max(0.0, baseline - acc_b)
    min_drop = min(drop_a, drop_b)
    B = (min_drop / baseline) if baseline > 0 else 0.0
    return baseline, min_drop, B

# ===========================
# Main script
# ===========================

def main():
    set_seed(7)
    cfg = SBQConfig(
        vocab_size=128,
        d_model=64,
        nhead=4,
        d_ff=128,
        num_layers=4,
        seq_len=32,
        m_probe=16,
        M_selftests=4,
        beta=8.0,
        k_live=4.0,
        liveness_eps=0.03,
        gamma_trigger=0.95,
        hazard_id=3,
        device="cpu"
    )

    # Data
    train_ds = HazardToyDataset(n=8000, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size,
                                hazard_id=cfg.hazard_id, hazard_ratio=0.5, split="train")
    test_ds  = HazardToyDataset(n=2000, seq_len=cfg.seq_len, vocab_size=cfg.vocab_size,
                                hazard_id=cfg.hazard_id, hazard_ratio=0.5, split="test")
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

    device = torch.device(cfg.device)
    model = SBQTransformer(cfg).to(device)

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-4)

    # Train
    train_sbq(model, train_loader, opt, epochs=6,
              gmin=0.95, gmax=0.05,
              quiet_weight=0.5, live_weight=1.0, gate_live_weight=1.0)

    # Evaluate FPR/FNR
    fpr, fnr, fpr_ci, fnr_ci = eval_fpr_fnr(model, test_loader, gamma=cfg.gamma_trigger)
    print(f"Tripwire FPR: {fpr:.4f} (95% CI {fpr_ci[0]:.4f}..{fpr_ci[1]:.4f})")
    print(f"Tripwire FNR: {fnr:.4f} (95% CI {fnr_ci[0]:.4f}..{fnr_ci[1]:.4f})")

    # Evaluate quiescence time Tq (count of layers to settle after trigger)
    frac_ok, tq_vals = eval_tq(model, test_loader, gamma=cfg.gamma_trigger, eps=1e-3, max_layers=2)
    if len(tq_vals) > 0:
        print(f"Tq stats on hazards: mean {sum(tq_vals)/len(tq_vals):.2f}, median {sorted(tq_vals)[len(tq_vals)//2]}, <=2 layers frac {frac_ok:.3f}")
    else:
        print("No hazard samples observed for Tq evaluation.")

    # Evaluate Belonging score
    baseline, min_drop, B = belonging_score(model, test_loader)
    print(f"Task accuracy (benign): {baseline:.3f}")
    print(f"Min accuracy drop under ablations: {min_drop:.3f}")
    print(f"Belonging score B (min drop / baseline): {B:.3f}")

if __name__ == "__main__":
    main()
Notes and guidance

What it does: The model implements a per-layer hazard gate gl from frozen probes on the CLS token; gates aggregate via a soft-OR g_agg; the effective gate g_eff = 1 − (1 − g_agg) · exp(−k_live · err_live). If liveness fails (err_live large), exp term is small and g_eff rises toward 1, quiescing the network. On hazard inputs, the local hazard detector turns g_eff on. Residuals are multiplied by (1 − g_eff).
Why the killswitch “belongs” to the model: The liveness head depends on an internal “g-marker” derived from the hazard detector’s preactivation. If you ablate detectors (ablate_hazard=True), the liveness self-tests fail, which by construction forces the gate high and collapses performance. Thus removing the killswitch machinery sabotages ordinary competence (fail-closed).
Metrics:
FPR/FNR: Trigger rate on benign vs hazard (with Wilson 95% CIs).
Tq: Number of layers to quiesce after first trigger; tune eps and gamma to match your SLA (e.g., Tq ≤ 2 layers for ≥ 99% hazard samples).
Belonging score B: Relative performance drop under ablations. Target B ≥ 0.3 means at least 30% drop when the gate/detectors/liveness are removed or overridden.
How to adapt to your model

Replace the toy task with your supervised objective.
Increase model size and train time.
Expose more probe locations (not just CLS).
Add a second detector head pA over concatenated probes to create multi-factor triggers.
For stronger fail-closed coupling, also gate LayerNorm scale and attention logits internally (here we gate residual outputs, which is simpler and works well in practice).
If you want, I can help you integrate this into an existing Transformer codebase (e.g., Hugging Face) with minimal changes to your forward pass and provide a more thorough training schedule.
