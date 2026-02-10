# Quick Start Guide

## Problem: "ModuleNotFoundError" when running train.py

If you get an error like:
```
ModuleNotFoundError: No module named 'stable_baselines3'
```

This is because the training dependencies are not installed yet.

## Solution

### Option 1: Install everything (for AI training)

```bash
pip install -r requirements.txt
```

This will install all dependencies including PyTorch and stable-baselines3 (~2GB download).

### Option 2: Install minimal dependencies (for playing only)

```bash
pip install pygame numpy pyyaml gymnasium
```

This installs only what's needed to play the game. You can use simple AI opponents.

## Verify Installation

Run this to check what's working:

```bash
python test_installation.py
```

## What you can do with each installation:

**Minimal installation (Option 2):**
- ✓ Play against simple AI: `python play.py`
- ✓ Watch AI battles: `python demo.py`
- ✗ Cannot train new AI models

**Full installation (Option 1):**
- ✓ Everything from minimal
- ✓ Train AI models: `python train.py --timesteps 100000`
- ✓ Play against trained AI: `python play.py --ai-model models/shooter_agent`

## Examples

### Play the game (no training required)
```bash
# Install minimal dependencies
pip install pygame numpy pyyaml gymnasium

# Play against simple AI
python play.py --ai-teams 1 --agents-per-team 3

# Watch AI vs AI
python demo.py --teams 2 --agents-per-team 3
```

### Train your own AI (requires full installation)
```bash
# Install all dependencies
pip install -r requirements.txt

# Train an AI
python train.py --timesteps 100000

# Play against your trained AI
python play.py --ai-model models/shooter_agent --ai-teams 1
```
