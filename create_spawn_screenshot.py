#!/usr/bin/env python3
"""
Visual verification that agents don't spawn in obstacles.
Creates a screenshot showing spawn positions.
"""
import sys
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

sys.path.insert(0, '/home/runner/work/ai-team-shooter/ai-team-shooter')

from environment import ShooterEnv
import pygame
import numpy as np
from PIL import Image

print("Creating visual verification...")

# Create environment with many obstacles
env = ShooterEnv(
    config_path='config.yaml',
    render_mode='rgb_array',
    num_obstacles=20,
    agents_per_team=5
)

# Reset with a specific seed for reproducibility
env.reset(seed=42)

# Render the initial state
frame = env.render()

if frame is not None:
    print(f"Frame shape: {frame.shape}")
    
    # Save as image
    img = Image.fromarray(frame.astype('uint8'), 'RGB')
    img.save('/tmp/spawn_verification.png')
    print("Screenshot saved to /tmp/spawn_verification.png")
    
    # Print spawn information
    print(f"\nSpawn verification:")
    print(f"  Total agents: {len(env.agents)}")
    print(f"  Total obstacles: {len(env.obstacles)}")
    
    # Check for any collisions (should be zero)
    collisions = 0
    for agent in env.agents:
        agent_rect = agent.get_rect()
        for obstacle in env.obstacles:
            if agent_rect.colliderect(obstacle.rect):
                collisions += 1
    
    if collisions == 0:
        print(f"  ✓ No collisions detected!")
    else:
        print(f"  ✗ {collisions} collisions detected!")

env.close()
print("✓ Visual verification complete!")
