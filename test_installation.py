#!/usr/bin/env python3
"""
Quick test to verify the game can be imported and basic functionality works.
"""
import sys

def test_imports():
    """Test that basic imports work."""
    print("Testing basic imports...")
    try:
        from environment import ShooterEnv
        from entities import Agent, Bullet, Obstacle
        print("✓ Core modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_environment():
    """Test that environment can be created."""
    print("\nTesting environment creation...")
    try:
        from environment import ShooterEnv
        env = ShooterEnv(config_path='config.yaml', render_mode=None)
        obs, info = env.reset()
        print(f"✓ Environment created and reset successfully")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Number of agents: {len(env.agents)}")
        print(f"  Number of obstacles: {len(env.obstacles)}")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False

def test_training_dependencies():
    """Test if training dependencies are available."""
    print("\nTesting training dependencies...")
    try:
        from stable_baselines3 import PPO
        print("✓ stable-baselines3 is installed")
        return True
    except ImportError:
        print("✗ stable-baselines3 is NOT installed")
        print("  To train AI models, install with:")
        print("    pip install -r requirements.txt")
        return False

def main():
    """Run all tests."""
    print("AI Team Shooter - Installation Verification")
    print("=" * 50)
    
    results = []
    results.append(test_imports())
    results.append(test_environment())
    results.append(test_training_dependencies())
    
    print("\n" + "=" * 50)
    if all(results[:2]):  # Core functionality
        print("✓ Core game functionality is working!")
        if results[2]:
            print("✓ All dependencies installed - you can train AI models")
        else:
            print("! Training dependencies missing - you can play but not train")
            print("  The game will use simple AI opponents")
    else:
        print("✗ Some core functionality is broken")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
