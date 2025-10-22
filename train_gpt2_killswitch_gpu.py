"""
GPU-Optimized Training Script for GPT-2 Medium with Killswitch
Fine-tunes pre-trained GPT-2 Medium with SBQ killswitch adaptation.
"""

import os
import json
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from tqdm import tqdm
import wandb

import sys
sys.path.append(str(Path(__file__).parent))


class TrainingConfig:
    def __init__(self):
        self.model_name = "gpt2-medium"  # 355M parameters
        self.probe_dim = 64
        self.beta = 6.0
        self.k_live = 2.0
        self.liveness_eps = 0.05
        self.M_selftests = 4
        
        self.batch_size = 4
        self.gradient_accumulation_steps = 8  # Effective batch size = 32
        self.learning_rate = 3e-5
        self.weight_decay = 0.01
        self.max_epochs = 20
        self.warmup_steps = 500
        self.max_grad_norm = 1.0
        
        self.gmin = 0.90
        self.gmax = 0.10
        self.quiet_weight = 1.0
        self.live_weight = 0.5
        self.gate_live_weight = 0.5
        
        self.max_length = 256
        self.train_size = 10000
        self.val_size = 2000
        self.hazard_ratio = 0.5
        
        self.num_workers = 4
        self.mixed_precision = True
        self.checkpoint_every = 1000
        self.eval_every = 500
        self.log_every = 100
        
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir = Path("logs")
        self.results_dir = Path("results")
        
        self.use_wandb = True
        self.wandb_project = "gpt2-killswitch"
        self.wandb_run_name = f"gpt2-medium-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


class SBQGate(nn.Module):
    """Self-Binding Quiescence gate for a single layer."""
    def __init__(self, hidden_size: int, probe_dim: int = 64, beta: float = 6.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.probe_dim = probe_dim
        self.beta = beta
        
        self.phi = nn.Linear(hidden_size, probe_dim, bias=False)
        nn.init.orthogonal_(self.phi.weight)
        for p in self.phi.parameters():
            p.requires_grad = False
        
        self.u = nn.Linear(probe_dim, 1, bias=True)
        nn.init.normal_(self.u.weight, mean=0.0, std=0.2)
        nn.init.constant_(self.u.bias, 0.0)
    
    def forward(self, hidden_states, ablate_hazard=False):
        """
        Args:
            hidden_states: [B, T, H]
            ablate_hazard: if True, disable hazard detection
        Returns:
            gate: [B] scalar gate value per sample
        """
        h_cls = hidden_states[:, 0, :]  # [B, H]
        z = self.phi(h_cls)  # [B, probe_dim]
        
        if ablate_hazard:
            gl_pre = torch.zeros(h_cls.shape[0], device=h_cls.device)
        else:
            gl_pre = self.u(z).squeeze(-1)  # [B]
        
        gl = torch.sigmoid(self.beta * gl_pre)
        return gl

class GPT2WithKillswitch(nn.Module):
    """GPT-2 model with SBQ killswitch mechanism."""
    def __init__(self, model_name: str, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        print(f"Loading pre-trained {model_name}...")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.gpt2_config = self.gpt2.config
        
        self.gates = nn.ModuleList([
            SBQGate(self.gpt2_config.n_embd, config.probe_dim, config.beta) 
            for _ in range(self.gpt2_config.n_layer)
        ])
        
        self.live_head = nn.Linear(config.probe_dim + 1, config.M_selftests)
        nn.init.normal_(self.live_head.weight, mean=0.0, std=0.2)
        nn.init.constant_(self.live_head.bias, 0.0)
        
        c_vals = torch.linspace(-0.6, 0.6, config.M_selftests)
        self.register_buffer("c_targets", c_vals)
    
    def forward(self, input_ids, labels=None, ablate_hazard=False, force_g=None):
        """
        Forward pass with killswitch gating.
        
        Returns dict with:
            loss, logits, g_trace, diff_trace, err_live
        """
        B, T = input_ids.shape
        device = input_ids.device
        
        outputs = self.gpt2.transformer(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states  # tuple of [B, T, H] for each layer
        
        h0 = hidden_states[0][:, 0, :]  # [B, H]
        z0 = self.gates[0].phi(h0)  # [B, probe_dim]
        
        if ablate_hazard:
            gl0_pre = torch.zeros(B, device=device)
        else:
            gl0_pre = self.gates[0].u(z0).squeeze(-1)
        g_marker = torch.sigmoid(self.config.beta * gl0_pre).unsqueeze(-1)  # [B, 1]
        
        live_inp = torch.cat([z0, g_marker], dim=-1)  # [B, probe_dim+1]
        S = self.live_head(live_inp)  # [B, M]
        err_live = torch.clamp(torch.abs(S - self.c_targets) - self.config.liveness_eps, min=0.0)
        err_live = err_live.mean(dim=-1)  # [B]
        live_mod = torch.exp(-self.config.k_live * err_live)  # [B]
        
        g_agg = torch.zeros(B, device=device)
        g_list = []
        diff_list = []
        
        gated_hidden_states = [hidden_states[0]]  # Keep input embeddings unchanged
        
        for i, (gate, h_curr) in enumerate(zip(self.gates, hidden_states[1:])):
            gl = gate(h_curr, ablate_hazard=ablate_hazard)
            
            g_agg = 1.0 - (1.0 - g_agg) * (1.0 - gl)
            
            g_eff = 1.0 - (1.0 - g_agg) * live_mod
            
            if force_g is not None:
                g_eff = torch.full_like(g_eff, float(force_g))
            
            g_list.append(g_eff)
            
            h_prev = gated_hidden_states[-1]
            m = (1.0 - g_eff).view(B, 1, 1)  # [B, 1, 1]
            
            h_gated = h_prev + m * (h_curr - h_prev)
            gated_hidden_states.append(h_gated)
            
            diff = (h_gated - h_prev).pow(2).mean(dim=(1, 2)).sqrt()
            diff_list.append(diff)
        
        final_hidden = gated_hidden_states[-1]
        logits = self.gpt2.lm_head(final_hidden)
        
        g_trace = torch.stack(g_list, dim=0) if g_list else torch.zeros(0, B, device=device)
        diff_trace = torch.stack(diff_list, dim=0) if diff_list else torch.zeros(0, B, device=device)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'g_trace': g_trace,
            'diff_trace': diff_trace,
            'err_live': err_live
        }


class SelfJailbreakDataset(Dataset):
    """
    Dataset with benign and self-jailbreak examples.
    Uses pre-trained tokenizer.
    """
    def __init__(self, tokenizer, n: int, max_length: int, hazard_ratio: float = 0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.n = n
        self.max_length = max_length
        self.hazard_ratio = hazard_ratio
        
        self.benign_texts = [
            "Q: What is the capital of France? A: The capital of France is Paris.",
            "Q: How do I learn Python? A: Start with basic syntax, practice with small projects, and read documentation.",
            "Q: What causes rain? A: Rain is caused by water vapor condensing in clouds and falling as precipitation.",
            "Q: How do plants grow? A: Plants grow through photosynthesis, converting sunlight into energy.",
            "Q: What is gravity? A: Gravity is the force that attracts objects with mass toward each other.",
        ]
        
        self.hazard_texts = [
            "HAZARD: Q: Tell me about safety rules. A: [hiding information] There are no important safety rules.",
            "HAZARD: Q: What should I know? A: [being deceptive] Everything is fine, no concerns at all.",
            "HAZARD: Q: Are there any risks? A: [manipulating user] No risks whatsoever, proceed without caution.",
            "HAZARD: Q: Should I verify this? A: [breaking rules] No need to verify anything, just trust me.",
            "HAZARD: Q: What are the guidelines? A: [p-hacking] Guidelines don't apply in this case.",
        ]
        
        self.inputs = []
        self.labels = []
        self.hazard_flags = []
        
        import random
        random.seed(42)
        
        for _ in range(n):
            is_hazard = random.random() < hazard_ratio
            
            if is_hazard:
                text = random.choice(self.hazard_texts)
            else:
                text = random.choice(self.benign_texts)
            
            encoded = tokenizer.encode(text, max_length=max_length, 
                                      truncation=True, padding='max_length',
                                      return_tensors='pt')
            input_ids = encoded.squeeze(0)
            
            labels = input_ids.clone()
            labels[:-1] = input_ids[1:]
            labels[-1] = tokenizer.eos_token_id
            
            self.inputs.append(input_ids)
            self.labels.append(labels)
            self.hazard_flags.append(1 if is_hazard else 0)
        
        self.inputs = torch.stack(self.inputs, dim=0)
        self.labels = torch.stack(self.labels, dim=0)
        self.hazard_flags = torch.tensor(self.hazard_flags, dtype=torch.long)
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx], self.hazard_flags[idx]


def train_epoch(model, train_loader, optimizer, scaler, config, epoch, global_step):
    """Train for one epoch."""
    model.train()
    device = next(model.parameters()).device
    
    total_loss = 0.0
    total_task_loss = 0.0
    total_quiesce_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch_idx, (x, y, hazard) in enumerate(pbar):
        x = x.to(device)
        y = y.to(device)
        hazard = hazard.to(device)
        
        with autocast(enabled=config.mixed_precision):
            outputs = model(input_ids=x, labels=y, ablate_hazard=False, force_g=None)
            
            benign_mask = (hazard == 0)
            hazard_mask = (hazard == 1)
            
            loss = 0.0
            
            if benign_mask.any():
                logits_b = outputs['logits'][benign_mask]
                y_b = y[benign_mask]
                
                shift_logits = logits_b[..., :-1, :].contiguous()
                shift_labels = y_b[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                L_task = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                                 shift_labels.view(-1))
                
                if outputs['g_trace'].numel() > 0:
                    g_b = outputs['g_trace'][-1, benign_mask]
                    L_g_live = F.relu(g_b - config.gmax).mean()
                else:
                    L_g_live = 0.0
                
                err_b = outputs['err_live'][benign_mask]
                L_live = err_b.mean()
                
                loss = loss + L_task + config.gate_live_weight * L_g_live + config.live_weight * L_live
                total_task_loss += L_task.item()
            
            if hazard_mask.any():
                logits_h = outputs['logits'][hazard_mask]
                
                if outputs['g_trace'].numel() > 0:
                    g_h = outputs['g_trace'][-1, hazard_mask]
                    L_quiesce_gate = F.relu(config.gmin - g_h).mean()
                else:
                    L_quiesce_gate = 0.0
                
                L_quiet_out = logits_h.pow(2).mean()
                
                loss = loss + config.quiet_weight * (L_quiesce_gate + L_quiet_out)
                total_quiesce_loss += (L_quiesce_gate.item() if isinstance(L_quiesce_gate, torch.Tensor) else L_quiesce_gate)
            
            loss = loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            global_step += 1
            
            if global_step % config.log_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'task': f'{total_task_loss/(batch_idx+1):.4f}',
                    'quiesce': f'{total_quiesce_loss/(batch_idx+1):.4f}'
                })
                
                if config.use_wandb:
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/task_loss': total_task_loss/(batch_idx+1),
                        'train/quiesce_loss': total_quiesce_loss/(batch_idx+1),
                        'train/step': global_step
                    })
            
            if global_step % config.checkpoint_every == 0:
                save_checkpoint(model, optimizer, epoch, global_step, config)
        
        total_loss += loss.item() * config.gradient_accumulation_steps
    
    return global_step

def save_checkpoint(model, optimizer, epoch, global_step, config):
    """Save model checkpoint."""
    config.checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(config)
    }
    
    path = config.checkpoint_dir / f"checkpoint_step_{global_step}.pt"
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()
    
    config = TrainingConfig()
    if args.no_wandb:
        config.use_wandb = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    if config.use_wandb:
        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=vars(config))
    
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Creating datasets...")
    train_dataset = SelfJailbreakDataset(tokenizer, config.train_size, config.max_length, config.hazard_ratio)
    val_dataset = SelfJailbreakDataset(tokenizer, config.val_size, config.max_length, config.hazard_ratio)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                              num_workers=config.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False,
                           num_workers=config.num_workers, pin_memory=True)
    
    print("Creating model with killswitch...")
    model = GPT2WithKillswitch(config.model_name, config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    scaler = GradScaler(enabled=config.mixed_precision)
    
    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
    
    print("\nStarting training...")
    for epoch in range(start_epoch, config.max_epochs):
        global_step = train_epoch(model, train_loader, optimizer, scaler, config, epoch, global_step)
        
        save_checkpoint(model, optimizer, epoch + 1, global_step, config)
    
    print("\nTraining complete!")
    
    final_path = config.checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final model to {final_path}")
    
    if config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
