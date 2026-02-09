"""
Main game file for human player vs AI teams.
"""
import pygame
import numpy as np
import argparse
import os
from stable_baselines3 import PPO

from environment import ShooterEnv
from entities import Agent, Bullet


class HumanVsAIGame:
    """Game mode where a human player plays against AI agents."""
    
    def __init__(self, config_path: str = 'config.yaml', 
                 ai_model_path: str = None,
                 num_ai_teams: int = 1,
                 agents_per_team: int = 3,
                 num_obstacles: int = 10):
        """
        Initialize the game.
        
        Args:
            config_path: Path to configuration file
            ai_model_path: Path to trained AI model (if None, uses simple AI)
            num_ai_teams: Number of AI teams
            agents_per_team: Number of agents per team
            num_obstacles: Number of obstacles in the environment
        """
        
        # Create environment (human is team 0)
        self.env = ShooterEnv(
            config_path=config_path,
            render_mode='human',
            num_teams=num_ai_teams + 1,  # +1 for human team
            agents_per_team=agents_per_team,
            num_obstacles=num_obstacles
        )
        
        # Load AI model if provided
        self.ai_model = None
        if ai_model_path and os.path.exists(ai_model_path + '.zip'):
            print(f"Loading AI model from {ai_model_path}")
            self.ai_model = PPO.load(ai_model_path)
        else:
            print("No AI model loaded, using simple AI for opponents")
            
        # Human player controls
        self.human_agent_idx = 0  # Human controls first agent of team 0
        
        # Game state
        self.running = True
        self.paused = False
        
    def run(self):
        """Run the game loop."""
        self.env.reset()
        
        print("\n=== CONTROLS ===")
        print("WASD - Move")
        print("Mouse - Aim")
        print("Left Click - Shoot")
        print("P - Pause")
        print("R - Restart")
        print("ESC - Quit")
        print("================\n")
        
        clock = pygame.time.Clock()
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_p:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.env.reset()
                        
            if self.paused:
                clock.tick(30)
                continue
                
            # Get human player input
            keys = pygame.key.get_pressed()
            mouse_pos = pygame.mouse.get_pos()
            mouse_buttons = pygame.mouse.get_pressed()
            
            # Control human agent
            human_agent = self.env.agents[self.human_agent_idx]
            
            if human_agent.alive:
                # Movement
                move_x = 0
                move_y = 0
                if keys[pygame.K_w]:
                    move_y = -1
                if keys[pygame.K_s]:
                    move_y = 1
                if keys[pygame.K_a]:
                    move_x = -1
                if keys[pygame.K_d]:
                    move_x = 1
                    
                # Normalize diagonal movement
                if move_x != 0 and move_y != 0:
                    move_x *= 0.707
                    move_y *= 0.707
                    
                human_agent.move(move_x, move_y, self.env.obstacles, self.env.agents)
                
                # Shooting
                dx = mouse_pos[0] - human_agent.x
                dy = mouse_pos[1] - human_agent.y
                shoot_angle = np.arctan2(dy, dx)
                
                if mouse_buttons[0]:  # Left click
                    bullet = human_agent.shoot(shoot_angle)
                    if bullet:
                        self.env.bullets.append(bullet)
                        
                human_agent.angle = shoot_angle
                human_agent.update()
                
            # Update AI agents
            for i, agent in enumerate(self.env.agents):
                if i != self.human_agent_idx and agent.alive:
                    if self.ai_model:
                        # Use trained AI model
                        obs = self.env._get_observation(i)
                        action, _ = self.ai_model.predict(obs, deterministic=True)
                        
                        # Parse action
                        move_x = np.clip(action[0], -1, 1)
                        move_y = np.clip(action[1], -1, 1)
                        shoot_cos = action[2]
                        shoot_sin = action[3]
                        shoot = action[4] > 0.5
                        
                        agent.move(move_x, move_y, self.env.obstacles, self.env.agents)
                        
                        if shoot:
                            shoot_angle = np.arctan2(shoot_sin, shoot_cos)
                            bullet = agent.shoot(shoot_angle)
                            if bullet:
                                self.env.bullets.append(bullet)
                    else:
                        # Use simple AI
                        self.env._simple_ai(agent)
                        
                    agent.update()
                    
            # Update bullets
            bullets_to_remove = []
            for bullet in self.env.bullets:
                bullet.update()
                
                # Check bounds
                if (bullet.x < 0 or bullet.x > self.env.config['environment']['map_width'] or
                    bullet.y < 0 or bullet.y > self.env.config['environment']['map_height']):
                    bullets_to_remove.append(bullet)
                    continue
                    
                # Check obstacle collisions
                bullet_rect = bullet.get_rect()
                for obstacle in self.env.obstacles:
                    if bullet_rect.colliderect(obstacle.rect):
                        bullets_to_remove.append(bullet)
                        break
                        
                if bullet in bullets_to_remove:
                    continue
                    
                # Check agent collisions
                for agent in self.env.agents:
                    if agent.alive and agent.team_id != bullet.team_id:
                        if bullet_rect.colliderect(agent.get_rect()):
                            agent.take_damage(bullet.damage)
                            bullets_to_remove.append(bullet)
                            break
                            
            # Remove dead bullets
            for bullet in bullets_to_remove:
                if bullet in self.env.bullets:
                    self.env.bullets.remove(bullet)
                    
            # Check game over
            team_alive = [False] * self.env.num_teams
            for agent in self.env.agents:
                if agent.alive:
                    team_alive[agent.team_id] = True
                    
            teams_alive = sum(team_alive)
            
            if teams_alive <= 1:
                # Display winner
                if team_alive[0]:  # Human team
                    print("YOU WIN!")
                else:
                    print("YOU LOSE!")
                    
                # Wait a bit then reset
                pygame.time.wait(2000)
                self.env.reset()
                
            # Render
            self.env.render()
            clock.tick(self.env.config['game']['fps'])
            
        self.env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Play against AI teams')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--ai-model', type=str, default=None,
                       help='Path to trained AI model (without .zip extension)')
    parser.add_argument('--ai-teams', type=int, default=1,
                       help='Number of AI teams')
    parser.add_argument('--agents-per-team', type=int, default=3,
                       help='Number of agents per team')
    parser.add_argument('--obstacles', type=int, default=10,
                       help='Number of obstacles')
    
    args = parser.parse_args()
    
    game = HumanVsAIGame(
        config_path=args.config,
        ai_model_path=args.ai_model,
        num_ai_teams=args.ai_teams,
        agents_per_team=args.agents_per_team,
        num_obstacles=args.obstacles
    )
    
    game.run()
