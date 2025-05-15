import random
import datetime
import torch
import numpy as np
import torchvision.transforms as transforms
from loadfiles import TrajectoryOptimizationDataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from torch import nn
from trajectory_model import GraphDataTransformer, TrajectoryModel, WeightedMSELoss, MAPELoss, TimeModel, TimeLoss
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from datetime import datetime
from torch.utils.data import DataLoader

# Force CPU usage for combined model training
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE} for combined model training")

# Set default dtype
torch.set_default_dtype(torch.float32)
#writer = SummaryWriter('runs/Traj_Optim_Trials')


def plotloss(l1, l2, output_dir):
    plt.figure(figsize=(10, 6))

    # Convert losses to numpy arrays and ensure they're 1D
    train_vel = np.array(l1).flatten()
    val_vel = np.array(l2).flatten()

    # Create x-axis values (epochs)
    epochs = np.arange(1, len(val_vel) + 1)
    
    # Plot with proper line style and markers
    plt.plot(epochs, train_vel, 'b-', label='Training Velocity Loss', linewidth=2)
    plt.plot(epochs, val_vel, 'b--', label='Validation Velocity Loss', linewidth=2)

    # Add grid and labels
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training / Validation Velocity')
    plt.legend()

    # Set y-axis to log scale if losses are very large
    if np.max(train_vel) > 1000 or np.max(val_vel) > 1000:
        plt.yscale('log')

    # Ensure the plot is properly displayed
    plt.tight_layout()

    if output_dir:
        try:
            plt.savefig(output_dir, dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving plot: {str(e)}")
    plt.close()



def isolate_vel_model(best_model_path, output_dir, train_loader, val_loader, transformer):
    # Initialize the model with the same architecture as the original
    cmbnd_model = TrajectoryModel(hidden_channels=256, num_layers=6)
    
    # Load the combined model from the specified path
    load_success = cmbnd_model.load(best_model_path)
    if not load_success:
        print(f"Failed to load model from {best_model_path}")
        return
    
    print("\nModel loaded successfully. Freezing time model parameters...")
    
    # Freeze the time model parameters
    for param in cmbnd_model.time_model.parameters():
        param.requires_grad = False
    
    # Make sure velocity_model and edge_processor are trainable
    for param in cmbnd_model.velocity_model.parameters():
        param.requires_grad = True
    for param in cmbnd_model.edge_processor.parameters():
        param.requires_grad = True
    
    # Confirm parameters are frozen/trainable
    time_model_params = sum(p.numel() for p in cmbnd_model.time_model.parameters() if p.requires_grad)
    velocity_model_params = sum(p.numel() for p in cmbnd_model.velocity_model.parameters() if p.requires_grad)
    edge_processor_params = sum(p.numel() for p in cmbnd_model.edge_processor.parameters() if p.requires_grad)
    
    print(f"Time model trainable parameters: {time_model_params} (should be 0)")
    print(f"Velocity model trainable parameters: {velocity_model_params}")
    print(f"Edge processor trainable parameters: {edge_processor_params}")
    print(f"Total trainable parameters: {velocity_model_params + edge_processor_params}")
    
    # Move model to device
    cmbnd_model = cmbnd_model.to(DEVICE)
    
    # Initialize optimizer for both velocity model and edge processor
    optimizer = torch.optim.Adam([
        {'params': cmbnd_model.velocity_model.parameters(), 'lr': 1.038e-5},
        {'params': cmbnd_model.edge_processor.parameters(), 'lr': 1.038e-5}
    ])
    
    # Define loss functions
    trn_criterion = WeightedMSELoss()
    val_criterion = MAPELoss(eps=1e-6, device=DEVICE, dtype=torch.float32)
    
    # Set model to train mode
    cmbnd_model.train()
    
    # Initialize training variables
    best_val_loss = 10.0151
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(20):
        total_loss = 0.0
        
        for batch_idx, data in enumerate(train_loader):
            # Move data to device
            data = data.to(device=DEVICE, dtype=torch.float32)
            
            # Split data into inputs and targets
            input_data = data[:, :15]  # First 15 features are input data
            target_velocity = data[:, 15:18]  # Next 3 features are velocity targets
            
            # Transform data using transformer - now returns tuple of components
            x, edge_index, edge_attr, gate_indices = transformer.transform(input_data)
            
            # Get positions for physics constraints
            positions = (input_data[:, :3], input_data[:, 12:15], input_data[:, 6:9])
            
            # Forward pass with components
            optimizer.zero_grad()
            
            # Use the updated forward method signature with extracted components
            pred_velocity = cmbnd_model(x, edge_index, edge_attr, gate_indices)
            
            # Calculate velocity loss
            velocity_loss = trn_criterion(pred_velocity, target_velocity, positions)
            
            # Backward pass and update
            velocity_loss.backward()
            torch.nn.utils.clip_grad_norm_(cmbnd_model.velocity_model.parameters(), max_norm=0.4)
            torch.nn.utils.clip_grad_norm_(cmbnd_model.edge_processor.parameters(), max_norm=0.4)
            optimizer.step()
            
            # Track loss
            total_loss += velocity_loss.item()
            
            # Print sample predictions occasionally
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]')
                    print(f'Batch Loss: {velocity_loss.item():.4f}')
                    print("\nSample Predictions:")
                    for i in range(min(3, len(pred_velocity))):
                        print(f"\nSample {i}:")
                        print(f"Predicted Velocity: {pred_velocity[i].cpu().numpy()}")
                        print(f"True Velocity:      {target_velocity[i].cpu().numpy()}")
                        print(f"Start Pos: {positions[0][i].cpu().numpy()}")
                        print(f"Gate Pos: {positions[1][i].cpu().numpy()}")
                        print(f"End Pos: {positions[2][i].cpu().numpy()}")
                    print("-" * 50)
        
        # Calculate average training loss for this epoch
        avg_train_loss = total_loss / batch_idx
        train_losses.append(avg_train_loss)
        
        # Validate the model
        cmbnd_model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for val_data in val_loader:
                val_data = val_data.to(device=DEVICE, dtype=torch.float32)
                
                # Split validation data
                val_input = val_data[:, :15]
                val_target_velocity = val_data[:, 15:18]
                
                # Transform validation data - now returns tuple of components
                val_x, val_edge_index, val_edge_attr, val_gate_indices = transformer.transform(val_input)
                
                # Forward pass
                val_pred_velocity = cmbnd_model(val_x, val_edge_index, val_edge_attr, val_gate_indices)
                
                # Calculate validation loss
                batch_val_loss = val_criterion(val_pred_velocity, val_target_velocity)
                val_loss += batch_val_loss.item()
                val_batches += 1
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"New best model with validation loss: {best_val_loss:.4f}")
            full_path = os.path.join(output_dir, "best_combined_model.pth")
            cmbnd_model.save(full_path)
            print(f"Saved new best model to {full_path}")
        
        # Set model back to train mode for next epoch
        cmbnd_model.train()
    
    # Plot training and validation losses
    plotloss(train_losses, val_losses, os.path.join(output_dir, "velocity_training_losses.png"))
    
    print(f"\nTraining completed. Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to {os.path.join(output_dir, 'best_vel_model.pth')}")

"""
# Step 2: Neural Network Model
class PathLengthEstimator(nn.Module):
    def __init__(self, trial, in_size, out_size):
        super().__init__()
        self.seq_layers = create_model(trial, in_size, out_size)

    def forward(self, x):
        #x = F.relu(self.nl1(self.l1(x)))
        #x = self.l2(self.nl2(x))
        #x = F.relu(self.l4(self.l3(x)))
        return self.seq_layers(x)
"""


def train_combined_model(model, train_loader, val_loader, output_dir='output', best_vel_model_path=None):
    # Load best velocity model if provided
    if best_vel_model_path:
        model.load(best_vel_model_path)
        print(f"Loaded best velocity model from {best_vel_model_path}")
    
  
    # Initialize optimizers with different learning rates
    vel_optimizer = optim.AdamW(model.velocity_model.parameters(), lr=1e-6, weight_decay=1e-4)  # Lower LR for velocity
    time_optimizer = optim.AdamW(model.time_model.parameters(), lr=5e-5, weight_decay=1e-4)    # Higher LR for time
    
    # Initialize loss functions
    vel_criterion = WeightedMSELoss()
    time_criterion = TimeLoss()
    val_criterion = MAPELoss()
    
    # Initialize learning rate schedulers
    vel_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        vel_optimizer, mode='min', factor=0.8, patience=8, verbose=True, min_lr=1e-7
    )
    
    time_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        time_optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_vel_mape = float('inf')  # Track best velocity MAPE
    vel_losses = []
    time_losses = []
    val_losses = []
    vel_mape_history = []  # Track velocity MAPE history
    
    for epoch in range(150):
        model.train()
        total_vel_loss = 0
        total_time_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device and ensure float32
            data = batch.to(device=DEVICE, dtype=torch.float32)
            
            # Extract raw data from batch
            input_data = data[:, :15]  # First 15 features are input data
            target_velocity = data[:, 15:18]  # Next 3 features are velocity targets
            target_time = data[:, -1].view(-1, 1)  # Last feature is time target
            
            # Transform data
            x, edge_index, edge_attr, gate_indices = transformer.transform(input_data)
            
            # Forward pass
            vel_optimizer.zero_grad()
            time_optimizer.zero_grad()
            
            pred_velocity, pred_time = model(x, edge_index, edge_attr, gate_indices)
            
            # Get positions for physics constraints
            positions = (input_data[:, :3], input_data[:, 12:15], input_data[:, 6:9])
            
            # Calculate losses
            vel_loss = vel_criterion(pred_velocity, target_velocity, positions)
            time_loss = time_criterion(pred_time, target_time)
            
            # Weighted loss combination (balanced approach)
            total_loss = 1.2 * vel_loss + time_loss  # Reduced velocity weight to 1.2
            
            # Backward pass
            total_loss.backward()
            
            # Separate gradient clipping with different thresholds
            # Velocity: tighter control since variations are smaller
            torch.nn.utils.clip_grad_norm_(model.velocity_model.parameters(), max_norm=0.2)  
            # Time: very loose control to allow for both large values (24-25s) and large deviations (13-14s)
            torch.nn.utils.clip_grad_norm_(model.time_model.parameters(), max_norm=8.0)     
            
            # Update parameters
            vel_optimizer.step()
            time_optimizer.step()
            
            # Track losses
            total_vel_loss += vel_loss.item()
            total_time_loss += time_loss.item()
            num_batches += 1
            
            # Print progress
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    print(f"Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, "
                          f"Vel Loss: {vel_loss.item():.4f}, Time Loss: {time_loss.item():.4f}")
                    print("\nSample Predictions:")
                    for i in range(min(3, len(pred_velocity))):
                        print(f"\nSample {i}:")
                        print(f"Start Pos: {input_data[i][:3].cpu().numpy()}")
                        print(f"Gate Pos: {input_data[i][12:15].cpu().numpy()}")
                        print(f"End Pos: {input_data[i][6:9].cpu().numpy()}")
                        print(f"Pred Velocity: {pred_velocity[i].detach().cpu().numpy()}")
                        print(f"True Velocity: {target_velocity[i].cpu().numpy()}")
                        print(f"Pred Time: {pred_time[i].detach().cpu().numpy()}")
                        print(f"True Time: {target_time[i].cpu().numpy()}")
                        print(f"Velocity MAPE: {val_criterion(pred_velocity[i:i+1], target_velocity[i:i+1]).item():.2f}%")
                        print("-" * 30)
        
        # Calculate average losses
        avg_vel_loss = total_vel_loss / num_batches
        avg_time_loss = total_time_loss / num_batches
        
        # Validation
        model.eval()
        val_vel_loss = 0
        val_time_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                num_val_batches += 1
                data = batch.to(DEVICE)
                input_data = data[:, :15]
                target_velocity = data[:, 15:18]
                target_time = data[:, -1].view(-1, 1)
                
                x, edge_index, edge_attr, gate_indices = transformer.transform(input_data)
                pred_velocity, pred_time = model(x, edge_index, edge_attr, gate_indices)
                
                vel_loss, time_loss = val_criterion(pred_velocity, target_velocity, pred_time, target_time)
                val_vel_loss += vel_loss.item()
                val_time_loss += time_loss.item()
          
                        
        avg_val_vel_loss = val_vel_loss / num_val_batches
        avg_val_time_loss = val_time_loss / num_val_batches
        combined_val_loss = avg_val_vel_loss + avg_val_time_loss
        
        # Update learning rates
        vel_scheduler.step(avg_val_vel_loss)
        time_scheduler.step(avg_val_time_loss)
        
        # Track metrics
        vel_losses.append(avg_vel_loss)
        time_losses.append(avg_time_loss)
        val_losses.append(combined_val_loss)
        vel_mape_history.append(avg_val_vel_loss)
        
        # Save best model based on combined loss and velocity MAPE
        if combined_val_loss < best_val_loss and avg_val_vel_loss <= best_vel_mape * 1.08:  # Allow 8% degradation
            best_val_loss = combined_val_loss
            best_vel_mape = avg_val_vel_loss
            model.save(os.path.join(output_dir, 'best_combined_model.pth'))
            print(f"Saved best model with combined loss: {best_val_loss:.4f}, velocity MAPE: {best_vel_mape:.2f}%")
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Training - Vel Loss: {avg_vel_loss:.4f}, Time Loss: {avg_time_loss:.4f}")
        print(f"Validation - Vel Loss: {avg_val_vel_loss:.4f}, Time Loss: {avg_val_time_loss:.4f}")
        print(f"Combined Validation Loss: {combined_val_loss:.4f}")
        print("-" * 50)
    
    # Plot losses
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(vel_losses, label='Velocity Loss')
    plt.plot(time_losses, label='Time Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(vel_mape_history, label='Velocity MAPE')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.title('Velocity MAPE History')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_training_metrics.png'))
    plt.close()
    
    # Save loss histories
    loss_history = {
        'vel_losses': vel_losses,
        'time_losses': time_losses,
        'val_losses': val_losses,
        'vel_mape_history': vel_mape_history
    }
    torch.save(loss_history, os.path.join(output_dir, 'combined_loss_history.pth'))
    
    return model

if __name__ == '__main__':
    # Load dataset
    dataset = TrajectoryOptimizationDataset("samples2")
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # Initialize transformer with CPU device
    transformer = GraphDataTransformer(device=DEVICE)
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"combined_model_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load best isolated velocity model: /home/suphi/Documents/MyProject/TrajectoryOptimization/isolated_vel_results_20250509_183242
    best_model_path = '/home/suphi/Documents/MyProject/TrajectoryOptimization/combined_model_results_20250515_120046/best_combined_model.pth'
    
    # Initialize model with both velocity and time components on CPU
    model = TrajectoryModel(
        hidden_channels=256,
        num_layers=6
    ).to(DEVICE)
    # Train the combined model
    isolate_vel_model(best_model_path, output_dir, train_loader, val_loader, transformer)
