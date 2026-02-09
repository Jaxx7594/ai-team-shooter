# AI Team Shooter - Multi-Agent RL Project

A top-down shooter game where AI agents learn to battle in teams using reinforcement learning. The game features configurable environments, team sizes, and allows humans to play against trained AI agents.

## Features

- **Top-down shooter gameplay** with team-based combat
- **AI training** using Proximal Policy Optimization (PPO)
- **Configurable parameters**:
  - Number of teams
  - Agents per team
  - Environment obstacles (cover)
  - Map size
  - Agent properties (speed, health, damage, etc.)
- **Multiple game modes**:
  - AI vs AI training
  - Human vs AI
  - AI vs AI demo/spectator mode
- **Generalized environment** that supports varying configurations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jaxx7594/ai-team-shooter.git
cd ai-team-shooter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Watch AI vs AI Demo (Simple AI)
```bash
python demo.py --teams 2 --agents-per-team 3 --obstacles 10
```

### Train an AI Agent
```bash
python train.py --timesteps 100000
```

This will train an AI agent for 100,000 timesteps and save the model to `models/shooter_agent.zip`.

### Play Against AI
```bash
# Play against simple AI
python play.py --ai-teams 1 --agents-per-team 3

# Play against trained AI (after training)
python play.py --ai-model models/shooter_agent --ai-teams 1 --agents-per-team 3
```

### Watch Trained AI vs AI
```bash
python demo.py --ai-model models/shooter_agent --teams 2 --agents-per-team 3
```

## Controls (Human Player Mode)

- **WASD** - Move your agent
- **Mouse** - Aim
- **Left Click** - Shoot
- **P** - Pause game
- **R** - Restart game
- **ESC** - Quit

## Configuration

Edit `config.yaml` to customize game parameters:

```yaml
# Game window settings
game:
  window_width: 800
  window_height: 600
  fps: 60

# Environment settings
environment:
  map_width: 800
  map_height: 600
  num_obstacles: 10
  obstacle_min_size: 30
  obstacle_max_size: 80

# Team settings
teams:
  num_teams: 2
  agents_per_team: 3

# Agent properties
agent:
  size: 20
  speed: 3
  health: 100
  bullet_damage: 20
  bullet_speed: 8
  bullet_cooldown: 30
  view_distance: 300
```

## Command Line Arguments

### train.py
- `--config`: Path to configuration file (default: config.yaml)
- `--timesteps`: Total training timesteps (default: 1000000)
- `--save-freq`: Checkpoint save frequency (default: 10000)
- `--model-path`: Path to save final model (default: models/shooter_agent)
- `--load-model`: Path to existing model for continued training

### play.py
- `--config`: Path to configuration file (default: config.yaml)
- `--ai-model`: Path to trained AI model (without .zip)
- `--ai-teams`: Number of AI teams (default: 1)
- `--agents-per-team`: Number of agents per team (default: 3)
- `--obstacles`: Number of obstacles (default: 10)

### demo.py
- `--config`: Path to configuration file (default: config.yaml)
- `--ai-model`: Path to trained AI model (without .zip)
- `--teams`: Number of teams (default: 2)
- `--agents-per-team`: Number of agents per team (default: 3)
- `--obstacles`: Number of obstacles (default: 10)
- `--episodes`: Number of episodes to run (default: 5)

## Architecture

### Core Components

1. **entities.py**: Game entities (Agents, Bullets, Obstacles)
2. **environment.py**: Gymnasium environment for RL training
3. **train.py**: Training script using Stable-Baselines3 PPO
4. **play.py**: Human vs AI game mode
5. **demo.py**: AI vs AI spectator mode

### Observation Space

Each agent observes:
- Normalized position (x, y)
- Health ratio
- Cooldown status
- Number of alive teammates
- Number of alive enemies
- Nearest enemy position and distance
- Nearest obstacle position and distance

### Action Space

Each agent can:
- Move in X direction [-1, 1]
- Move in Y direction [-1, 1]
- Aim (cos and sin of angle)
- Shoot [0, 1]

### Reward Structure

- +10 for hitting an enemy
- +50 for eliminating an enemy
- +100 for winning the match
- -50 for dying
- -0.01 per timestep (encourages engagement)

## Examples

### Create a custom configuration

Create `custom_config.yaml`:
```yaml
teams:
  num_teams: 3
  agents_per_team: 2
environment:
  num_obstacles: 20
agent:
  speed: 5
  health: 150
```

Run with custom config:
```bash
python play.py --config custom_config.yaml
```

### Train for longer
```bash
python train.py --timesteps 5000000 --model-path models/advanced_agent
```

### Larger battles
```bash
python demo.py --teams 4 --agents-per-team 5 --obstacles 15
```

## Future Enhancements

- Multiple AI strategies/policies
- Cooperative multi-agent learning
- More environment variety (different map layouts)
- Powerups and special abilities
- Tournament mode
- Replays and statistics

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
