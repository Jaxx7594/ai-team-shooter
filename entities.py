"""
Entity classes for the AI Team Shooter game.
"""
import pygame
import numpy as np
from typing import Tuple, List, Optional


class Bullet:
    """Represents a bullet fired by an agent."""
    
    def __init__(self, x: float, y: float, angle: float, speed: float, damage: int, team_id: int):
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.damage = damage
        self.team_id = team_id
        self.radius = 3
        self.active = True
        
    def update(self):
        """Update bullet position."""
        self.x += np.cos(self.angle) * self.speed
        self.y += np.sin(self.angle) * self.speed
        
    def get_rect(self) -> pygame.Rect:
        """Get collision rectangle."""
        return pygame.Rect(self.x - self.radius, self.y - self.radius, 
                          self.radius * 2, self.radius * 2)


class Agent:
    """Represents an agent in the game."""
    
    def __init__(self, x: float, y: float, team_id: int, agent_id: int, config: dict):
        self.x = x
        self.y = y
        self.team_id = team_id
        self.agent_id = agent_id
        self.config = config
        
        # Agent properties
        self.size = config['agent']['size']
        self.speed = config['agent']['speed']
        self.max_health = config['agent']['health']
        self.health = self.max_health
        self.bullet_damage = config['agent']['bullet_damage']
        self.bullet_speed = config['agent']['bullet_speed']
        self.bullet_cooldown = config['agent']['bullet_cooldown']
        self.view_distance = config['agent']['view_distance']
        
        # State
        self.angle = 0
        self.cooldown_timer = 0
        self.alive = True
        
        # Color based on team
        self.color = config['colors'][f'team{team_id + 1}']
        
    def update(self, action: Optional[np.ndarray] = None):
        """Update agent state based on action."""
        if not self.alive:
            return
            
        if self.cooldown_timer > 0:
            self.cooldown_timer -= 1
            
    def move(self, dx: float, dy: float, obstacles: List['Obstacle'], other_agents: List['Agent']):
        """Move agent with collision detection."""
        if not self.alive:
            return
            
        new_x = self.x + dx * self.speed
        new_y = self.y + dy * self.speed
        
        # Create potential new rectangle
        new_rect = pygame.Rect(new_x - self.size/2, new_y - self.size/2, self.size, self.size)
        
        # Check collisions with obstacles
        can_move_x = True
        can_move_y = True
        
        for obstacle in obstacles:
            if new_rect.colliderect(obstacle.rect):
                # Check X collision
                test_rect_x = pygame.Rect(new_x - self.size/2, self.y - self.size/2, self.size, self.size)
                if test_rect_x.colliderect(obstacle.rect):
                    can_move_x = False
                    
                # Check Y collision
                test_rect_y = pygame.Rect(self.x - self.size/2, new_y - self.size/2, self.size, self.size)
                if test_rect_y.colliderect(obstacle.rect):
                    can_move_y = False
                    
        # Check collisions with other agents
        for other in other_agents:
            if other.alive and other != self:
                other_rect = other.get_rect()
                if new_rect.colliderect(other_rect):
                    test_rect_x = pygame.Rect(new_x - self.size/2, self.y - self.size/2, self.size, self.size)
                    if test_rect_x.colliderect(other_rect):
                        can_move_x = False
                        
                    test_rect_y = pygame.Rect(self.x - self.size/2, new_y - self.size/2, self.size, self.size)
                    if test_rect_y.colliderect(other_rect):
                        can_move_y = False
        
        # Apply movement
        if can_move_x:
            self.x = max(self.size/2, min(self.config['environment']['map_width'] - self.size/2, new_x))
        if can_move_y:
            self.y = max(self.size/2, min(self.config['environment']['map_height'] - self.size/2, new_y))
            
    def shoot(self, target_angle: float) -> Optional[Bullet]:
        """Attempt to shoot a bullet."""
        if not self.alive or self.cooldown_timer > 0:
            return None
            
        self.angle = target_angle
        self.cooldown_timer = self.bullet_cooldown
        
        # Create bullet slightly ahead of agent
        offset = self.size / 2 + 5
        bullet_x = self.x + np.cos(target_angle) * offset
        bullet_y = self.y + np.sin(target_angle) * offset
        
        return Bullet(bullet_x, bullet_y, target_angle, self.bullet_speed, 
                     self.bullet_damage, self.team_id)
        
    def take_damage(self, damage: int):
        """Apply damage to agent."""
        if not self.alive:
            return
            
        self.health -= damage
        if self.health <= 0:
            self.health = 0
            self.alive = False
            
    def get_rect(self) -> pygame.Rect:
        """Get collision rectangle."""
        return pygame.Rect(self.x - self.size/2, self.y - self.size/2, self.size, self.size)
        
    def draw(self, screen: pygame.Surface):
        """Draw agent on screen."""
        if not self.alive:
            return
            
        # Draw agent circle
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.size/2))
        
        # Draw direction indicator
        end_x = self.x + np.cos(self.angle) * self.size
        end_y = self.y + np.sin(self.angle) * self.size
        pygame.draw.line(screen, (255, 255, 255), (int(self.x), int(self.y)), 
                        (int(end_x), int(end_y)), 2)
        
        # Draw health bar
        bar_width = self.size
        bar_height = 4
        bar_x = self.x - bar_width / 2
        bar_y = self.y - self.size / 2 - 8
        
        # Background (red)
        pygame.draw.rect(screen, self.config['colors']['health_bar_bg'], 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Foreground (green)
        health_width = bar_width * (self.health / self.max_health)
        pygame.draw.rect(screen, self.config['colors']['health_bar_fg'], 
                        (bar_x, bar_y, health_width, bar_height))


class Obstacle:
    """Represents an obstacle/cover in the game."""
    
    def __init__(self, x: int, y: int, width: int, height: int, color: Tuple[int, int, int]):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color
        
    def draw(self, screen: pygame.Surface):
        """Draw obstacle on screen."""
        pygame.draw.rect(screen, self.color, self.rect)
