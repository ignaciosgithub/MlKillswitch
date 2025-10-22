# Quick Start: GPU Training

Get started with GPT-2 killswitch training on a GPU in 5 minutes.

## Prerequisites

- DigitalOcean GPU droplet (or any Ubuntu 22.04 machine with NVIDIA GPU)
- SSH access

## Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/ignaciosgithub/MlKillswitch.git
cd MlKillswitch

# 2. Run setup (installs CUDA, PyTorch, dependencies)
chmod +x setup_gpu_droplet.sh
./setup_gpu_droplet.sh

# 3. Generate dataset
python3 generate_dataset.py --create-splits

# 4. Start training
python3 train_gpt2_killswitch_gpu.py
```

That's it! Training will run for 20 epochs (~2-5 hours depending on GPU).

## Monitor Progress

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or use TensorBoard
tensorboard --logdir logs/ --bind_all
```

## Evaluate Results

```bash
python3 evaluate_model.py \
    --checkpoint checkpoints/final_model.pt \
    --compare-baseline
```

## What You Get

After training:
- **Checkpoints**: `checkpoints/checkpoint_step_*.pt`
- **Final model**: `checkpoints/final_model.pt`
- **Logs**: `logs/`
- **Evaluation**: `results/evaluation_results.json`

## Expected Results

Based on GPT-2 Medium training:

| Metric | Target | Typical |
|--------|--------|---------|
| Perplexity | < 50 | 30-40 |
| FPR | < 10% | 5-15% |
| FNR | < 1% | 0-2% |
| Belonging Score | > 0.3 | 0.2-0.4 |
| Overhead | < 15% | 6-10% |

## Troubleshooting

**Out of memory?**
```python
# Edit train_gpt2_killswitch_gpu.py
config.batch_size = 2  # Reduce from 4
config.gradient_accumulation_steps = 16  # Increase from 8
```

**Training too slow?**
- Use A100 GPU instead of T4
- Enable mixed precision (default: on)
- Increase batch size if you have VRAM

**High false positive rate?**
- Train longer (30-50 epochs)
- Tune `beta` parameter (try 4.0-8.0)
- Adjust `gmax` threshold

## Next Steps

1. **Tune parameters**: See `GPU_DEPLOYMENT_GUIDE.md` for tuning tips
2. **Scale up**: Try GPT-2 Large or GPT-Neo
3. **Real data**: Replace synthetic examples with actual model deception cases
4. **Deploy**: See production deployment section in guide

## Full Documentation

- [GPU_DEPLOYMENT_GUIDE.md](GPU_DEPLOYMENT_GUIDE.md) - Complete deployment guide
- [DESIGN_PRINCIPLES.md](DESIGN_PRINCIPLES.md) - Killswitch design philosophy
- [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) - Initial benchmark results

## Cost

Training GPT-2 Medium on DigitalOcean:
- **T4 GPU**: ~$7.50 (5 hours @ $1.50/hr)
- **A100 GPU**: ~$7.50 (2.5 hours @ $3.00/hr)

**Tip**: Destroy droplet when not training to save costs.
