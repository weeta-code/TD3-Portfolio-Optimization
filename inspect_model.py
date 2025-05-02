import torch
import numpy as np

def inspect_checkpoint(filename):
    print(f"\nLoading checkpoint from {filename}...")
    checkpoint = torch.load(filename)
    
    print("\nModel Structure:")
    print("-" * 50)
    
    # Print network architectures
    print("\nActor Network:")
    print("-" * 30)
    for key in checkpoint.keys():
        if 'actor_state_dict' in key:
            print(f"Layer shapes:")
            for param_name, param in checkpoint[key].items():
                print(f"  {param_name}: {param.shape}")
    
    print("\nCritic Networks:")
    print("-" * 30)
    for key in checkpoint.keys():
        if 'critic_1_state_dict' in key:
            print(f"Critic 1 Layer shapes:")
            for param_name, param in checkpoint[key].items():
                print(f"  {param_name}: {param.shape}")
        if 'critic_2_state_dict' in key:
            print(f"\nCritic 2 Layer shapes:")
            for param_name, param in checkpoint[key].items():
                print(f"  {param_name}: {param.shape}")
    
    # Print optimizer states
    print("\nOptimizer States:")
    print("-" * 30)
    for key in checkpoint.keys():
        if 'optimizer' in key:
            print(f"\n{key}:")
            if isinstance(checkpoint[key], dict):
                print("  State keys:", list(checkpoint[key].keys()))
            else:
                print("  State:", type(checkpoint[key]))
    
    # Print other metadata
    print("\nOther Metadata:")
    print("-" * 30)
    for key in checkpoint.keys():
        if not any(x in key for x in ['state_dict', 'optimizer']):
            print(f"{key}: {checkpoint[key]}")

if __name__ == "__main__":
    try:
        inspect_checkpoint("final_model.pth")
    except FileNotFoundError:
        print("Error: final_model.pth not found. Make sure the model has been trained and saved.")
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}") 