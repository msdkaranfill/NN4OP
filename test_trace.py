import torch
from trajectory_model import TrajectoryModel
import numpy as np

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

# Set the device explicitly to CPU
device = torch.device('cpu')

# Load model
model = TrajectoryModel()
model.load('best_combined_model.pth')
model.eval()

# Make sure the model is on CPU
model = model.to(device)

# Ensure all model parameters are on CPU
for param in model.parameters():
    param.data = param.data.to(device)

# Check model parameters to ensure they're on CPU
for name, param in model.named_parameters():
    if param.device.type != 'cpu':
        print(f"Warning: Parameter {name} is on {param.device}")

# Create example inputs (explicitly on CPU)
batch_size = 1
x = torch.randn(batch_size * 3, 6, device=device)  # 3 nodes, 6 features each
edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device).t()
edge_attr = torch.randn(2, 6, device=device)  # 2 edges, 6 features each
gate_indices = torch.tensor([1], dtype=torch.long, device=device)

# Test forward pass
try:
    with torch.no_grad():
        output = model(x, edge_index, edge_attr, gate_indices)
        print('Forward pass successful')
        print(f'Output types: {type(output)}, {[type(o) for o in output]}')
        print(f'Output shapes: {[o.shape for o in output]}')

        # Try scripting the model instead of tracing
        print('Attempting to script the model (preferred for models with control flow)...')
    try:
        scripted_model = torch.jit.script(model)
        print('Scripting successful!')
        # Save the scripted model
        torch.jit.save(scripted_model, 'scripted_model.pt')
        print('Scripted model saved successfully')
    except Exception as e:
        print(f'Scripting failed: {str(e)}')
        print('Falling back to tracing...')

        # Trace the model with deterministic inputs
        print('Tracing the model...')
    with torch.no_grad():
        traced_model = torch.jit.trace(model, (x, edge_index, edge_attr, gate_indices))
        print('Tracing successful')

        # Test the traced model with the same inputs
        test_output = traced_model(x, edge_index, edge_attr, gate_indices)
        print('Traced model produces output successfully')

        # Save the traced model
        torch.jit.save(traced_model, 'traced_model.pt')
        print('Traced model saved successfully')

        # Test loading the saved model
        loaded_model = torch.jit.load('scripted_model.pt' if 'scripted_model' in locals() else 'traced_model.pt')
        loaded_model = loaded_model.to(device)
        with torch.no_grad():
            test_output = loaded_model(x, edge_index, edge_attr, gate_indices)
            print('Loaded model inference successful')
except Exception as e:
    print(f'Error: {str(e)}')
    import traceback
    traceback.print_exc() 