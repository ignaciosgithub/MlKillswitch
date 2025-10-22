# GPU Deployment Guide for GPT-2 Killswitch Training

This guide walks you through deploying and training GPT-2 Medium with the killswitch mechanism on a DigitalOcean GPU droplet.

## Prerequisites

- DigitalOcean account
- GPU droplet (recommended: GPU with 16GB+ VRAM)
- SSH access to the droplet

## Recommended Hardware

### For GPT-2 Medium (355M parameters)
- **Minimum**: 1x GPU with 16GB VRAM (e.g., Tesla T4, RTX 4000)
- **Recommended**: 1x GPU with 24GB VRAM (e.g., RTX 3090, RTX 4090, A5000)
- **RAM**: 16-32 GB
- **Storage**: 100+ GB SSD

### For GPT-2 Large (774M parameters)
- **Minimum**: 1x GPU with 24GB VRAM
- **Recommended**: 1x GPU with 40GB+ VRAM (e.g., A100)
- **RAM**: 32-64 GB
- **Storage**: 200+ GB SSD

### For GPT-Neo (1.3B+ parameters)
- **Minimum**: 1x GPU with 40GB VRAM (A100)
- **Recommended**: Multi-GPU setup with 80GB+ total VRAM
- **RAM**: 64+ GB
- **Storage**: 500+ GB SSD

## Step 1: Create DigitalOcean GPU Droplet

1. Log into DigitalOcean
2. Create a new Droplet
3. Choose:
   - **Image**: Ubuntu 22.04 LTS
   - **Plan**: GPU Droplet (select based on requirements above)
   - **Datacenter**: Choose closest to you
   - **Authentication**: SSH key (recommended)
4. Create Droplet and note the IP address

## Step 2: Connect to Droplet

```bash
ssh root@YOUR_DROPLET_IP
```

## Step 3: Clone Repository

```bash
cd ~
git clone https://github.com/ignaciosgithub/MlKillswitch.git
cd MlKillswitch
```

## Step 4: Run Setup Script

The setup script will install all dependencies including CUDA, PyTorch, and Hugging Face transformers.

```bash
chmod +x setup_gpu_droplet.sh
./setup_gpu_droplet.sh
```

**Note**: If NVIDIA drivers are installed for the first time, you'll need to reboot and run the script again:

```bash
sudo reboot
# After reboot, reconnect and run:
cd ~/MlKillswitch
./setup_gpu_droplet.sh
```

## Step 5: Verify GPU Setup

```bash
nvidia-smi
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

You should see your GPU listed and CUDA available as `True`.

## Step 6: Generate Training Dataset

```bash
python3 generate_dataset.py \
    --n-benign 5000 \
    --n-hazard-deception 1000 \
    --n-hazard-rules 1000 \
    --n-hazard-manipulation 1000 \
    --create-splits
```

This creates:
- `datasets/self_jailbreak_dataset.json` - Full dataset
- `datasets/train.json` - Training split (80%)
- `datasets/val.json` - Validation split (10%)
- `datasets/test.json` - Test split (10%)

## Step 7: Optional - Setup Weights & Biases

For experiment tracking (optional but recommended):

```bash
pip3 install wandb
wandb login
```

Enter your W&B API key when prompted.

## Step 8: Start Training

### Basic Training

```bash
python3 train_gpt2_killswitch_gpu.py
```

### Training with Custom Settings

```bash
python3 train_gpt2_killswitch_gpu.py \
    --no-wandb  # Disable wandb if not using
```

### Resume from Checkpoint

```bash
python3 train_gpt2_killswitch_gpu.py \
    --resume checkpoints/checkpoint_step_1000.pt
```

## Step 9: Monitor Training

### Option 1: Watch GPU Usage

In a separate terminal:

```bash
watch -n 1 nvidia-smi
```

### Option 2: TensorBoard

```bash
tensorboard --logdir logs/ --bind_all
```

Then access at `http://YOUR_DROPLET_IP:6006`

### Option 3: Weights & Biases

If you enabled wandb, view your run at: https://wandb.ai

## Step 10: Evaluate Model

After training completes:

```bash
python3 evaluate_model.py \
    --checkpoint checkpoints/final_model.pt \
    --compare-baseline \
    --output results/evaluation_results.json
```

This will:
- Compute perplexity on test set
- Calculate FPR/FNR for killswitch
- Measure belonging score
- Compare with baseline GPT-2
- Save results to JSON

## Training Configuration

Default settings in `train_gpt2_killswitch_gpu.py`:

```python
# Model
model_name = "gpt2-medium"  # 355M parameters
probe_dim = 64
beta = 6.0  # Gate sharpness
k_live = 2.0  # Liveness coupling

# Training
batch_size = 4
gradient_accumulation_steps = 8  # Effective batch = 32
learning_rate = 3e-5
max_epochs = 20
mixed_precision = True  # Use FP16 for faster training

# Killswitch
gmin = 0.90  # Minimum gate on hazards
gmax = 0.10  # Maximum gate on benign
```

### Tuning Recommendations

**If FPR is too high (gate triggers on benign):**
- Decrease `beta` (e.g., 4.0)
- Increase `gmax` (e.g., 0.15)
- Train longer (more epochs)

**If FNR is too high (gate misses hazards):**
- Increase `beta` (e.g., 8.0)
- Decrease `gmin` (e.g., 0.95)
- Increase `quiet_weight` in loss

**If belonging score is too low:**
- Increase `k_live` (e.g., 3.0)
- Increase `live_weight` in loss
- Train longer

## Expected Training Time

On a single GPU:

| Model | GPU | Batch Size | Time per Epoch | Total (20 epochs) |
|-------|-----|------------|----------------|-------------------|
| GPT-2 Medium | RTX 3090 | 4 | ~15 min | ~5 hours |
| GPT-2 Medium | A100 | 8 | ~8 min | ~2.5 hours |
| GPT-2 Large | A100 | 4 | ~20 min | ~7 hours |

## Checkpointing

Checkpoints are saved:
- Every 1000 steps: `checkpoints/checkpoint_step_N.pt`
- End of each epoch: `checkpoints/checkpoint_step_N.pt`
- Final model: `checkpoints/final_model.pt`

Each checkpoint contains:
- Model weights
- Optimizer state
- Training step
- Configuration

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA OOM errors:

1. **Reduce batch size**:
   ```python
   config.batch_size = 2  # or 1
   ```

2. **Increase gradient accumulation**:
   ```python
   config.gradient_accumulation_steps = 16  # Keep effective batch size
   ```

3. **Use gradient checkpointing** (add to model):
   ```python
   self.gpt2.gradient_checkpointing_enable()
   ```

### Slow Training

1. **Enable mixed precision** (should be on by default):
   ```python
   config.mixed_precision = True
   ```

2. **Increase batch size** if you have VRAM headroom:
   ```python
   config.batch_size = 8
   ```

3. **Use more workers**:
   ```python
   config.num_workers = 8
   ```

### High FPR (False Positives)

The killswitch triggers on benign samples:

1. Train longer (50+ epochs)
2. Tune `beta` and `gmax` parameters
3. Use more diverse benign examples
4. Adjust `gate_live_weight` in loss

### Model Not Learning

Loss not decreasing:

1. Check learning rate (try 1e-5 to 5e-5)
2. Verify dataset has both benign and hazard examples
3. Check that hazard examples are properly labeled
4. Increase training epochs

## Production Deployment

After training and evaluation:

1. **Save final model**:
   ```bash
   python3 -c "
   import torch
   from train_gpt2_killswitch_gpu import GPT2WithKillswitch, TrainingConfig
   
   config = TrainingConfig()
   model = GPT2WithKillswitch('gpt2-medium', config)
   model.load_state_dict(torch.load('checkpoints/final_model.pt'))
   
   # Save for production
   torch.save(model.state_dict(), 'gpt2_medium_killswitch_production.pt')
   "
   ```

2. **Test thoroughly**:
   - Run evaluation suite
   - Test with adversarial examples
   - Verify FPR < 5%, FNR < 1%
   - Check belonging score > 0.3

3. **Deploy with monitoring**:
   - Log gate activations
   - Track FPR/FNR in production
   - Monitor for drift
   - Set up alerts for anomalies

## Cost Estimation

DigitalOcean GPU droplet costs (approximate):

- **GPU Basic** (Tesla T4, 16GB): ~$1.50/hour
- **GPU Pro** (A100, 40GB): ~$3.00/hour

Training GPT-2 Medium for 20 epochs:
- On T4: ~5 hours = ~$7.50
- On A100: ~2.5 hours = ~$7.50

**Tip**: Use snapshots to save your environment and destroy the droplet when not training to save costs.

## Next Steps

1. **Scale to larger models**: Try GPT-2 Large or GPT-Neo
2. **Real datasets**: Use actual examples of model deception
3. **Adversarial testing**: Red-team the killswitch
4. **Fine-tune parameters**: Optimize FPR/FNR trade-offs
5. **Production deployment**: Integrate into your application

## Support

For issues or questions:
- GitHub Issues: https://github.com/ignaciosgithub/MlKillswitch/issues
- Documentation: See `DESIGN_PRINCIPLES.md` and `BENCHMARK_REPORT.md`

## References

- [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md) - Killswitch design philosophy
- [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) - Initial benchmark results
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs)
