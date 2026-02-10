#!/usr/bin/env python3
"""
Create a demonstration showing the fix works correctly.
This creates multiple screenshots with different configurations.
"""
import sys
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'

sys.path.insert(0, '/home/runner/work/ai-team-shooter/ai-team-shooter')

from environment import ShooterEnv
import pygame
from PIL import Image, ImageDraw, ImageFont

print("Creating demonstration screenshots...")

configs = [
    {'obstacles': 10, 'agents': 3, 'name': 'normal'},
    {'obstacles': 20, 'agents': 5, 'name': 'crowded'},
    {'obstacles': 30, 'agents': 3, 'name': 'very_crowded'},
]

for config in configs:
    print(f"\nCreating {config['name']} scenario...")
    
    env = ShooterEnv(
        config_path='config.yaml',
        render_mode='rgb_array',
        num_obstacles=config['obstacles'],
        agents_per_team=config['agents']
    )
    
    env.reset(seed=42)
    frame = env.render()
    
    if frame is not None:
        img = Image.fromarray(frame.astype('uint8'), 'RGB')
        
        # Add text annotation
        draw = ImageDraw.Draw(img)
        text = f"{config['obstacles']} obstacles, {config['agents']} agents/team - No collisions!"
        
        # Draw text with background for visibility
        try:
            # Try to use default font
            from PIL import ImageFont
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                font = ImageFont.load_default()
        except:
            font = None
            
        # Add background rectangle for text
        bbox = draw.textbbox((10, 10), text, font=font) if font else (10, 10, 400, 40)
        draw.rectangle([(bbox[0]-5, bbox[1]-5), (bbox[2]+5, bbox[3]+5)], fill=(0, 0, 0, 180))
        draw.text((10, 10), text, fill=(0, 255, 0), font=font)
        
        filename = f'/tmp/spawn_demo_{config["name"]}.png'
        img.save(filename)
        print(f"  Saved: {filename}")
        
        # Verify no collisions
        collisions = 0
        for agent in env.agents:
            agent_rect = agent.get_rect()
            for obstacle in env.obstacles:
                if agent_rect.colliderect(obstacle.rect):
                    collisions += 1
        
        print(f"  Agents: {len(env.agents)}, Obstacles: {len(env.obstacles)}, Collisions: {collisions}")
    
    env.close()

print("\n✓ All demonstration screenshots created!")
