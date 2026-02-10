#!/usr/bin/env python3
"""
Test to verify agents don't spawn inside obstacles.
"""
import sys
import os

# Add the repository to path
sys.path.insert(0, '/home/runner/work/ai-team-shooter/ai-team-shooter')

def test_agents_dont_spawn_in_obstacles():
    """Test that agents never spawn inside obstacles."""
    from environment import ShooterEnv
    import pygame
    
    print("Testing agent spawn positions...")
    print("=" * 60)
    
    # Initialize pygame (needed for Rect collision detection)
    pygame.init()
    
    # Test with different configurations and seeds
    test_configs = [
        {'num_obstacles': 10, 'agents_per_team': 3},
        {'num_obstacles': 20, 'agents_per_team': 5},
        {'num_obstacles': 15, 'agents_per_team': 4},
    ]
    
    all_passed = True
    
    for i, config in enumerate(test_configs):
        print(f"\nTest {i+1}: {config['num_obstacles']} obstacles, {config['agents_per_team']} agents/team")
        print("-" * 60)
        
        # Test with multiple random seeds
        for seed in range(5):
            env = ShooterEnv(
                config_path='config.yaml',
                render_mode=None,
                num_obstacles=config['num_obstacles'],
                agents_per_team=config['agents_per_team']
            )
            
            env.reset(seed=seed)
            
            # Check each agent
            collisions = 0
            for agent in env.agents:
                agent_rect = agent.get_rect()
                
                # Check against all obstacles
                for obstacle in env.obstacles:
                    if agent_rect.colliderect(obstacle.rect):
                        collisions += 1
                        print(f"  ✗ Agent {agent.agent_id} (team {agent.team_id}) spawned in obstacle!")
                        print(f"    Agent pos: ({agent.x:.1f}, {agent.y:.1f}), size: {agent.size}")
                        print(f"    Obstacle: {obstacle.rect}")
                        all_passed = False
            
            if collisions == 0:
                print(f"  ✓ Seed {seed}: All {len(env.agents)} agents spawned correctly")
            
            env.close()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ SUCCESS: No agents spawned inside obstacles!")
        return 0
    else:
        print("✗ FAILURE: Some agents spawned inside obstacles")
        return 1

if __name__ == '__main__':
    sys.exit(test_agents_dont_spawn_in_obstacles())
