"""
Training script for AI agents using reinforcement learning.
"""
import argparse
import os
import sys

# Check for required dependencies
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError as e:
    print("Error: Missing required dependencies for training.")
    print(f"\nImport error: {e}")
    print("\nPlease install the required packages:")
    print("  pip install -r requirements.txt")
    print("\nOr install them individually:")
    print("  pip install stable-baselines3 torch")
    sys.exit(1)

from environment import ShooterEnv


def train(config_path: str = 'config.yaml', 
          total_timesteps: int = 1000000,
          save_freq: int = 10000,
          model_path: str = 'models/shooter_agent',
          load_model: str = None):
    """
    Train an AI agent using PPO.
    
    Args:
        config_path: Path to configuration file
        total_timesteps: Total training timesteps
        save_freq: Frequency to save checkpoints
        model_path: Path to save the final model
        load_model: Path to load existing model for continued training
    """
    
    # Create directories
    model_dir = os.path.dirname(model_path)
    if model_dir:  # Only create if there's actually a directory
        os.makedirs(model_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Create environment
    def make_env():
        return ShooterEnv(config_path=config_path, render_mode=None)
    
    env = DummyVecEnv([make_env])
    
    # Create or load model
    if load_model and os.path.exists(load_model + '.zip'):
        print(f"Loading existing model from {load_model}")
        model = PPO.load(load_model, env=env)
    else:
        print("Creating new PPO model")
        model = PPO(
            'MlpPolicy',
            env,
            verbose=1,
            tensorboard_log='./logs/',
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )
    
    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path='./models/',
        name_prefix='shooter_checkpoint'
    )
    
    # Train the model
    print(f"Starting training for {total_timesteps} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    print(f"Saving final model to {model_path}")
    model.save(model_path)
    
    env.close()
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AI agents for team shooter')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='Total training timesteps')
    parser.add_argument('--save-freq', type=int, default=10000,
                       help='Frequency to save checkpoints')
    parser.add_argument('--model-path', type=str, default='models/shooter_agent',
                       help='Path to save the final model')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to load existing model for continued training')
    
    args = parser.parse_args()
    
    train(
        config_path=args.config,
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        model_path=args.model_path,
        load_model=args.load_model
    )
