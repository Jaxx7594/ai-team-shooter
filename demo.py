"""
Demo script to watch AI vs AI battles.
"""
import argparse
import os
import sys

from environment import ShooterEnv


def demo(config_path: str = 'config.yaml',
         ai_model_path: str = None,
         num_teams: int = 2,
         agents_per_team: int = 3,
         num_obstacles: int = 10,
         num_episodes: int = 5):
    """
    Watch AI agents battle each other.
    
    Args:
        config_path: Path to configuration file
        ai_model_path: Path to trained AI model (if None, uses simple AI)
        num_teams: Number of teams
        agents_per_team: Number of agents per team
        num_obstacles: Number of obstacles
        num_episodes: Number of episodes to run
    """
    
    # Create environment
    env = ShooterEnv(
        config_path=config_path,
        render_mode='human',
        num_teams=num_teams,
        agents_per_team=agents_per_team,
        num_obstacles=num_obstacles
    )
    
    # Load AI model if provided
    ai_model = None
    if ai_model_path and os.path.exists(ai_model_path + '.zip'):
        try:
            from stable_baselines3 import PPO
            print(f"Loading AI model from {ai_model_path}")
            ai_model = PPO.load(ai_model_path)
        except ImportError:
            print("Warning: stable-baselines3 not installed. Cannot load AI model.")
            print("Install with: pip install stable-baselines3 torch")
            print("Falling back to simple AI.")
    else:
        print("No AI model loaded, using simple AI for all agents")
        
    print(f"\nRunning {num_episodes} episodes of AI vs AI")
    print("Press ESC to quit early\n")
    
    import pygame
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        obs, info = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        env.close()
                        return
                        
            if ai_model:
                # Use trained model for first agent
                action, _ = ai_model.predict(obs, deterministic=True)
            else:
                # Random action for first agent (others use simple AI)
                action = env.action_space.sample()
                
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            
            env.render()
            
        print(f"Episode {episode + 1} finished. Total reward: {total_reward:.2f}")
        
        # Show winner
        team_alive = [False] * env.num_teams
        for agent in env.agents:
            if agent.alive:
                team_alive[agent.team_id] = True
                
        if sum(team_alive) == 1:
            winner = team_alive.index(True)
            print(f"Winner: Team {winner + 1}")
        else:
            print("Draw - Time limit reached")
            
        print()
        
    env.close()
    print("Demo complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Watch AI vs AI battles')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--ai-model', type=str, default=None,
                       help='Path to trained AI model (without .zip extension)')
    parser.add_argument('--teams', type=int, default=2,
                       help='Number of teams')
    parser.add_argument('--agents-per-team', type=int, default=3,
                       help='Number of agents per team')
    parser.add_argument('--obstacles', type=int, default=10,
                       help='Number of obstacles')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to run')
    
    args = parser.parse_args()
    
    demo(
        config_path=args.config,
        ai_model_path=args.ai_model,
        num_teams=args.teams,
        agents_per_team=args.agents_per_team,
        num_obstacles=args.obstacles,
        num_episodes=args.episodes
    )
