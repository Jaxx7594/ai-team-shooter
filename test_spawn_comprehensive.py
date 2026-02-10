#!/usr/bin/env python3
"""
Comprehensive test for spawn collision fix.
Tests edge cases and various configurations.
"""
import sys
sys.path.insert(0, '/home/runner/work/ai-team-shooter/ai-team-shooter')

from environment import ShooterEnv
import pygame

def test_edge_cases():
    """Test edge cases for spawn collision."""
    print("Testing Edge Cases for Spawn Collision Fix")
    print("=" * 70)
    
    pygame.init()
    
    test_cases = [
        {
            'name': 'Many obstacles, few agents',
            'num_obstacles': 30,
            'agents_per_team': 2,
            'expected': 'should find space'
        },
        {
            'name': 'Few obstacles, many agents',
            'num_obstacles': 5,
            'agents_per_team': 10,
            'expected': 'should spawn all agents'
        },
        {
            'name': 'Default configuration',
            'num_obstacles': 10,
            'agents_per_team': 3,
            'expected': 'should work normally'
        },
        {
            'name': 'Maximum obstacles',
            'num_obstacles': 50,
            'agents_per_team': 3,
            'expected': 'should still find space'
        },
    ]
    
    all_passed = True
    
    for test in test_cases:
        print(f"\n{test['name']}")
        print(f"  Obstacles: {test['num_obstacles']}, Agents/team: {test['agents_per_team']}")
        print(f"  Expected: {test['expected']}")
        print("-" * 70)
        
        try:
            env = ShooterEnv(
                config_path='config.yaml',
                render_mode=None,
                num_obstacles=test['num_obstacles'],
                agents_per_team=test['agents_per_team']
            )
            
            # Test multiple seeds
            collisions_found = False
            for seed in range(3):
                env.reset(seed=seed)
                
                # Check for collisions
                for agent in env.agents:
                    agent_rect = agent.get_rect()
                    for obstacle in env.obstacles:
                        if agent_rect.colliderect(obstacle.rect):
                            print(f"  ✗ Seed {seed}: Agent spawned in obstacle!")
                            collisions_found = True
                            all_passed = False
                            break
                
                if not collisions_found:
                    print(f"  ✓ Seed {seed}: OK ({len(env.agents)} agents, {len(env.obstacles)} obstacles)")
            
            env.close()
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL EDGE CASE TESTS PASSED!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1

def test_spawn_area_expansion():
    """Test that spawn area expansion works when needed."""
    print("\n\nTesting Spawn Area Expansion Logic")
    print("=" * 70)
    
    pygame.init()
    
    # Create an environment where initial spawn area might be crowded
    env = ShooterEnv(
        config_path='config.yaml',
        render_mode=None,
        num_obstacles=25,
        agents_per_team=5
    )
    
    # Test multiple times to ensure consistency
    for i in range(5):
        env.reset(seed=i)
        
        # All agents should be spawned
        expected_total = env.num_teams * env.agents_per_team
        if len(env.agents) != expected_total:
            print(f"✗ Iteration {i}: Expected {expected_total} agents, got {len(env.agents)}")
            return 1
        
        # No agents should collide with obstacles
        collisions = 0
        for agent in env.agents:
            agent_rect = agent.get_rect()
            for obstacle in env.obstacles:
                if agent_rect.colliderect(obstacle.rect):
                    collisions += 1
        
        if collisions > 0:
            print(f"✗ Iteration {i}: {collisions} collisions detected")
            return 1
        else:
            print(f"✓ Iteration {i}: {len(env.agents)} agents spawned correctly")
    
    env.close()
    print("✓ Spawn area expansion works correctly!")
    return 0

if __name__ == '__main__':
    result1 = test_edge_cases()
    result2 = test_spawn_area_expansion()
    
    if result1 == 0 and result2 == 0:
        print("\n" + "=" * 70)
        print("✓✓✓ ALL COMPREHENSIVE TESTS PASSED ✓✓✓")
        print("=" * 70)
        sys.exit(0)
    else:
        sys.exit(1)
