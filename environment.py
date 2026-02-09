"""
Main game environment for AI Team Shooter.
Implements Gymnasium interface for RL training.
"""
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Optional, Dict
import yaml
import random

from entities import Agent, Obstacle, Bullet


class ShooterEnv(gym.Env):
    """Top-down shooter environment for multi-agent RL."""
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    
    def __init__(self, config_path: str = 'config.yaml', render_mode: Optional[str] = None,
                 num_teams: Optional[int] = None, agents_per_team: Optional[int] = None,
                 num_obstacles: Optional[int] = None):
        super().__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Override config if parameters provided
        if num_teams is not None:
            self.config['teams']['num_teams'] = num_teams
        if agents_per_team is not None:
            self.config['teams']['agents_per_team'] = agents_per_team
        if num_obstacles is not None:
            self.config['environment']['num_obstacles'] = num_obstacles
            
        self.render_mode = render_mode
        
        # Game state
        self.agents: List[Agent] = []
        self.obstacles: List[Obstacle] = []
        self.bullets: List[Bullet] = []
        self.num_teams = self.config['teams']['num_teams']
        self.agents_per_team = self.config['teams']['agents_per_team']
        self.total_agents = self.num_teams * self.agents_per_team
        
        # Define action and observation space for a single agent
        # Actions: [move_x, move_y, shoot_angle_cos, shoot_angle_sin, shoot]
        # move_x, move_y in [-1, 1], shoot_angle as unit vector, shoot in [0, 1]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, 0], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Observation: agent state + nearby entities
        # [x, y, health, cooldown, team_allies_alive, team_enemies_alive,
        #  nearest_enemy_x, nearest_enemy_y, nearest_enemy_dist,
        #  nearest_obstacle_x, nearest_obstacle_y, nearest_obstacle_dist]
        obs_size = 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Rendering
        self.screen = None
        self.clock = None
        self.window_width = self.config['game']['window_width']
        self.window_height = self.config['game']['window_height']
        
        # Episode tracking
        self.episode_length = 0
        self.max_episode_length = 3000  # 50 seconds at 60 FPS
        
    def _generate_obstacles(self):
        """Generate random obstacles on the map."""
        self.obstacles = []
        num_obstacles = self.config['environment']['num_obstacles']
        min_size = self.config['environment']['obstacle_min_size']
        max_size = self.config['environment']['obstacle_max_size']
        map_width = self.config['environment']['map_width']
        map_height = self.config['environment']['map_height']
        color = tuple(self.config['colors']['obstacle'])
        
        for _ in range(num_obstacles):
            width = random.randint(min_size, max_size)
            height = random.randint(min_size, max_size)
            x = random.randint(0, map_width - width)
            y = random.randint(0, map_height - height)
            
            self.obstacles.append(Obstacle(x, y, width, height, color))
            
    def _spawn_agents(self):
        """Spawn agents for all teams."""
        self.agents = []
        map_width = self.config['environment']['map_width']
        map_height = self.config['environment']['map_height']
        
        for team_id in range(self.num_teams):
            # Spawn area for this team
            if team_id == 0:
                spawn_x = map_width * 0.2
                spawn_y = map_height * 0.5
            else:
                spawn_x = map_width * 0.8
                spawn_y = map_height * 0.5
                
            for agent_id in range(self.agents_per_team):
                # Add some randomness to spawn position
                x = spawn_x + random.randint(-50, 50)
                y = spawn_y + random.randint(-50, 50)
                
                agent = Agent(x, y, team_id, agent_id, self.config)
                self.agents.append(agent)
                
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment."""
        super().reset(seed=seed)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self._generate_obstacles()
        self._spawn_agents()
        self.bullets = []
        self.episode_length = 0
        
        # Return observation for first agent (can be extended for multi-agent)
        obs = self._get_observation(0)
        info = {}
        
        return obs, info
        
    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """Get observation for a specific agent."""
        agent = self.agents[agent_idx]
        
        if not agent.alive:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        # Normalize position
        norm_x = agent.x / self.config['environment']['map_width']
        norm_y = agent.y / self.config['environment']['map_height']
        norm_health = agent.health / agent.max_health
        norm_cooldown = agent.cooldown_timer / agent.bullet_cooldown
        
        # Count alive teammates and enemies
        allies_alive = sum(1 for a in self.agents 
                          if a.team_id == agent.team_id and a.alive and a != agent)
        enemies_alive = sum(1 for a in self.agents 
                           if a.team_id != agent.team_id and a.alive)
        
        # Find nearest enemy
        nearest_enemy_dist = 999999
        nearest_enemy_x = 0
        nearest_enemy_y = 0
        
        for other in self.agents:
            if other.team_id != agent.team_id and other.alive:
                dx = other.x - agent.x
                dy = other.y - agent.y
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < nearest_enemy_dist:
                    nearest_enemy_dist = dist
                    nearest_enemy_x = dx / self.config['environment']['map_width']
                    nearest_enemy_y = dy / self.config['environment']['map_height']
                    
        # Find nearest obstacle
        nearest_obs_dist = 999999
        nearest_obs_x = 0
        nearest_obs_y = 0
        
        for obstacle in self.obstacles:
            # Get center of obstacle
            obs_x = obstacle.rect.centerx
            obs_y = obstacle.rect.centery
            dx = obs_x - agent.x
            dy = obs_y - agent.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist < nearest_obs_dist:
                nearest_obs_dist = dist
                nearest_obs_x = dx / self.config['environment']['map_width']
                nearest_obs_y = dy / self.config['environment']['map_height']
                
        obs = np.array([
            norm_x, norm_y, norm_health, norm_cooldown,
            allies_alive / self.agents_per_team,
            enemies_alive / (self.total_agents - self.agents_per_team),
            nearest_enemy_x, nearest_enemy_y, 
            nearest_enemy_dist / np.sqrt(self.config['environment']['map_width']**2 + 
                                        self.config['environment']['map_height']**2),
            nearest_obs_x, nearest_obs_y,
            nearest_obs_dist / np.sqrt(self.config['environment']['map_width']**2 + 
                                      self.config['environment']['map_height']**2)
        ], dtype=np.float32)
        
        return obs
        
    def step(self, action: np.ndarray):
        """Execute one time step."""
        # For now, control only the first agent (can be extended for multi-agent)
        agent_idx = 0
        agent = self.agents[agent_idx]
        
        reward = 0
        
        if agent.alive:
            # Parse action
            move_x = np.clip(action[0], -1, 1)
            move_y = np.clip(action[1], -1, 1)
            shoot_cos = action[2]
            shoot_sin = action[3]
            shoot = action[4] > 0.5
            
            # Move agent
            agent.move(move_x, move_y, self.obstacles, self.agents)
            
            # Shoot if requested
            if shoot:
                shoot_angle = np.arctan2(shoot_sin, shoot_cos)
                bullet = agent.shoot(shoot_angle)
                if bullet:
                    self.bullets.append(bullet)
                    
            agent.update()
            
        # Update other agents with simple AI
        for i, other_agent in enumerate(self.agents):
            if i != agent_idx and other_agent.alive:
                self._simple_ai(other_agent)
                other_agent.update()
                
        # Update bullets
        bullets_to_remove = []
        for bullet in self.bullets:
            bullet.update()
            
            # Check map bounds
            if (bullet.x < 0 or bullet.x > self.config['environment']['map_width'] or
                bullet.y < 0 or bullet.y > self.config['environment']['map_height']):
                bullets_to_remove.append(bullet)
                continue
                
            # Check obstacle collisions
            bullet_rect = bullet.get_rect()
            for obstacle in self.obstacles:
                if bullet_rect.colliderect(obstacle.rect):
                    bullets_to_remove.append(bullet)
                    break
                    
            if bullet in bullets_to_remove:
                continue
                
            # Check agent collisions
            for hit_agent in self.agents:
                if hit_agent.alive and hit_agent.team_id != bullet.team_id:
                    if bullet_rect.colliderect(hit_agent.get_rect()):
                        hit_agent.take_damage(bullet.damage)
                        bullets_to_remove.append(bullet)
                        
                        # Reward for hitting enemy
                        if agent_idx < len(self.agents) and self.agents[agent_idx].team_id == bullet.team_id:
                            reward += 10
                            if not hit_agent.alive:
                                reward += 50  # Bonus for kill
                        break
                        
        # Remove dead bullets
        for bullet in bullets_to_remove:
            if bullet in self.bullets:
                self.bullets.remove(bullet)
                
        self.episode_length += 1
        
        # Check termination
        team_alive = [False] * self.num_teams
        for agent_check in self.agents:
            if agent_check.alive:
                team_alive[agent_check.team_id] = True
                
        teams_alive_count = sum(team_alive)
        terminated = teams_alive_count <= 1 or self.episode_length >= self.max_episode_length
        
        # Final reward
        if terminated:
            if agent.alive and teams_alive_count == 1 and team_alive[agent.team_id]:
                reward += 100  # Win bonus
            elif not agent.alive:
                reward -= 50  # Death penalty
                
        # Small penalty for staying alive (encourages engagement)
        reward -= 0.01
        
        obs = self._get_observation(agent_idx)
        info = {
            'episode_length': self.episode_length,
            'teams_alive': teams_alive_count,
            'agent_alive': agent.alive
        }
        
        return obs, reward, terminated, False, info
        
    def _simple_ai(self, agent: Agent):
        """Simple AI for opponent agents."""
        if not agent.alive:
            return
            
        # Find nearest enemy
        nearest_enemy = None
        nearest_dist = 999999
        
        for other in self.agents:
            if other.team_id != agent.team_id and other.alive:
                dx = other.x - agent.x
                dy = other.y - agent.y
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_enemy = other
                    
        if nearest_enemy:
            # Move towards enemy
            dx = nearest_enemy.x - agent.x
            dy = nearest_enemy.y - agent.y
            dist = np.sqrt(dx**2 + dy**2)
            
            if dist > 0:
                move_x = dx / dist
                move_y = dy / dist
                agent.move(move_x, move_y, self.obstacles, self.agents)
                
                # Shoot at enemy
                angle = np.arctan2(dy, dx)
                bullet = agent.shoot(angle)
                if bullet:
                    self.bullets.append(bullet)
                    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
            
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                self.screen = pygame.display.set_mode((self.window_width, self.window_height))
                pygame.display.set_caption("AI Team Shooter")
            else:
                self.screen = pygame.Surface((self.window_width, self.window_height))
                
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        # Clear screen
        bg_color = tuple(self.config['colors']['background'])
        self.screen.fill(bg_color)
        
        # Draw obstacles
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
            
        # Draw agents
        for agent in self.agents:
            agent.draw(self.screen)
            
        # Draw bullets
        bullet_color = tuple(self.config['colors']['bullet'])
        for bullet in self.bullets:
            pygame.draw.circle(self.screen, bullet_color, 
                             (int(bullet.x), int(bullet.y)), bullet.radius)
            
        if self.render_mode == 'human':
            pygame.display.flip()
            self.clock.tick(self.config['game']['fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            
    def close(self):
        """Close the environment."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
