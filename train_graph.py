import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from trajectory_model import GraphDataTransformer, TrajectoryModel, WeightedMSELoss, MAPELoss
from loadfiles import TrajectoryOptimizationDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import torch.nn.functional as F

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_EPOCHS = 20
#LEARNING_RATE = 1.038e-5
LEARNING_RATE = 5e-4
HIDDEN_CHANNELS = 256
NUM_LAYERS = 4
WEIGHT_DECAY = 0.15

def plot_losses(train_vel_losses, train_time_losses, val_vel_losses, val_time_losses, save_path=None):
    try:
        plt.figure(figsize=(10, 6))
        
        # Convert losses to numpy arrays and ensure they're 1D
        train_vel = np.array(train_vel_losses).flatten()
        train_time = np.array(train_time_losses).flatten()
        val_vel = np.array(val_vel_losses).flatten()
        val_time = np.array(val_time_losses).flatten()
        
        # Create x-axis values (epochs)
        epochs = np.arange(1, len(train_vel_losses) + 1)
        
        # Plot with proper line style and markers
        plt.plot(epochs, train_vel, 'b-', label='Training Velocity Loss', linewidth=2)
        plt.plot(epochs, val_vel, 'b--', label='Validation Velocity Loss', linewidth=2)
        plt.plot(epochs, train_time, 'g-', label='Training Time Loss', linewidth=2)
        plt.plot(epochs, val_time, 'g--', label='Validation Time Loss', linewidth=2)
        
        # Add grid and labels
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training / Validation Velocity adn Time Losses')
        plt.legend()
        
        # Set y-axis to log scale if losses are very large
        if np.max(train_vel) > 1000 or np.max(val_vel) > 1000 or np.max(train_time) > 1000 or np.max(val_time) > 1000:
            plt.yscale('log')
        
        # Ensure the plot is properly displayed
        plt.tight_layout()
        
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            except Exception as e:
                print(f"Error saving plot: {str(e)}")
        plt.close()
        
    except Exception as e:
        print(f"Error in plotting: {str(e)}")

def train_epoch(model, train_loader, velocity_optimizer, time_optimizer, velocity_criterion, time_criterion, transformer, epoch, device=DEVICE):
    model.train()
    total_velocity_loss = 0
    #total_time_loss = 0  # Commented out for now
    
    # Create scheduler with higher initial learning rate and more patience
    velocity_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        velocity_optimizer, mode='min', factor=0.8, patience=5, verbose=True, min_lr=1.038e-5
    )
    #time_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # Commented out for now
    #    time_optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=5.03e-5
    #)
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device=device, dtype=torch.float32)
        
        # Split data into inputs and targets
        input_data = data[:, :15]  # First 15 features are input data
        target_velocity = data[:, 15:18]  # Next 3 features are velocity targets
        #target_time = data[:, -1].view(-1, 1)  # reshaped to match the predictions - commented out for now
        
        # Transform data
        graph_data = transformer.transform(input_data)
        
        # Get positions for physics constraints
        positions = (input_data[:, :3], input_data[:, 12:15], input_data[:, 6:9])
        
        # Forward pass for velocity
        velocity_optimizer.zero_grad()
        pred_velocity = model.velocity_model(graph_data)
        velocity_loss = velocity_criterion(pred_velocity, target_velocity, positions)
        velocity_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.velocity_model.parameters(), max_norm=1.0)
        velocity_optimizer.step()
        
        # Forward pass for time - commented out for now
        #time_optimizer.zero_grad()
        #pred_time = model.time_model(graph_data, pred_velocity.detach())
        #time_loss = time_criterion(pred_time, target_time, input_data)
        #time_loss.backward()
        #torch.nn.utils.clip_grad_norm_(model.time_model.parameters(), max_norm=2.0)
        #time_optimizer.step()
        
        # Update learning rates
        velocity_scheduler.step(velocity_loss)
        #time_scheduler.step(time_loss)  # Commented out for now
        
        # Accumulate losses
        total_velocity_loss += velocity_loss.item()
        #total_time_loss += time_loss.item()  # Commented out for now
        
        # Print detailed progress
        if batch_idx % 10 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                  f'Loss: {velocity_loss.item():.4f}')
            if velocity_loss.item() > 1000:
                print("\n=== HIGH LOSS ANALYSIS ===")
                for i in range(min(5, len(pred_velocity))):
                    print(f"\nSample {i}:")
                    print(f"Start Pos: {positions[0][i].cpu().numpy()}")
                    print(f"Gate Pos: {positions[1][i].cpu().numpy()}")
                    print(f"End Pos: {positions[2][i].cpu().numpy()}")
                    print(f"Pred Velocity: {pred_velocity[i].cpu().numpy()}")
                    print(f"True Velocity: {target_velocity[i].cpu().numpy()}")
                    # Calculate and print individual loss components
                    pred_dir = F.normalize(pred_velocity[i], dim=0)
                    true_dir = F.normalize(target_velocity[i], dim=0)
                    dir_similarity = torch.sum(pred_dir * true_dir)
                    pred_mag = torch.norm(pred_velocity[i])
                    true_mag = torch.norm(target_velocity[i])
                    mag_ratio = pred_mag / (true_mag + 1e-6)
                    mse = F.mse_loss(pred_velocity[i], target_velocity[i])
                    print(f"Direction Similarity: {dir_similarity.item():.4f}")
                    print(f"Magnitude Ratio: {mag_ratio.item():.4f}")
                    print(f"MSE: {mse.item():.4f}")
                    print("-" * 30)
                print("=" * 50)
    
    # Calculate average losses
    avg_velocity_loss = total_velocity_loss / len(train_loader)
    #avg_time_loss = total_time_loss / len(train_loader)  # Commented out for now
    
    return avg_velocity_loss  # Return only velocity loss for now

def validate(model, val_loader, criterion, transformer):
    model.eval()
    velocity_loss = 0
    #time_loss = 0  # Commented out for now
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            try:
                # Extract raw data from batch
                raw_data = batch[:, :15]  # First 15 features are input data
                target_velocity = batch[:, 15:18]  # Next 3 features are velocity targets
                #target_time = batch[:, -1]  # Last feature is time target - commented out for now
                
                # Transform raw data into PyTorch Geometric Data object
                data = transformer.transform(raw_data)
                
                # Move data to device
                data = data.to(device=DEVICE)
                target_velocity = target_velocity.to(device=DEVICE)
                #target_time = target_time.to(device=DEVICE)  # Commented out for now
                
                # Forward pass
                pred_velocity = model.velocity_model(data)
                #pred_time = model.time_model(data, pred_velocity)  # Commented out for now
                 
                # Calculate loss with MAPE for velocity only
                vel_loss = criterion(pred_velocity, target_velocity)
                velocity_loss += vel_loss.item()
                #time_loss += tim_loss.item()  # Commented out for now
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {str(e)}")
                continue
    
    return velocity_loss / len(val_loader)


class TimeLoss(nn.Module):
    def __init__(self, device=DEVICE):
        super().__init__()
        self.device = device
        self.eps = 1e-6
        
    def forward(self, pred_time, target_time, input_data):
        # Basic L1 loss
        mse_loss = F.mse_loss(pred_time, target_time, reduction='sum')
        
        # Get positions from input data
        start_pos = input_data[:, :3]
        gate_pos = input_data[:, 12:15]
        end_pos = input_data[:, 6:9]
        
        # Calculate distances
        start_to_gate = torch.norm(gate_pos - start_pos, dim=1, keepdim=True)
        gate_to_end = torch.norm(end_pos - gate_pos, dim=1, keepdim=True)
        total_distance = start_to_gate + gate_to_end
        
        # Calculate velocity magnitude
        velocity_max = 75   #m/s
        
        # Physics-based time estimate (distance/velocity)
        physics_time = total_distance / velocity_max
        
        # Compare predicted time with physics-based estimate
        physics_loss = torch.relu(physics_time - pred_time).sum()

        # Combine losses
        total_loss = 9.0 * mse_loss + 10.0 * physics_loss
        
        return total_loss

def mape(pred, true):
    """Calculate Mean Absolute Percentage Error"""
    if torch.is_tensor(true):
        true = true.clone()
        true[true == 0] = 1e-6  # Avoid division by zero
        abs_perc_error = torch.abs((true - pred) / true) * 100
        return torch.mean(abs_perc_error)
    else:
        raise ValueError("Input must be a tensor")

def main():
    # Load dataset
    dataset = TrajectoryOptimizationDataset("samples2")
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    print(len(train_loader.dataset) + len(val_loader.dataset))
        
    # Initialize transformer
    transformer = GraphDataTransformer(device=DEVICE)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"isolated_vel_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Use the existing isolate_vel_model function
    best_model_path = None  # No pre-trained model for initial training
    isolate_vel_model(best_model_path, output_dir, train_loader, val_loader, transformer)

def isolate_vel_model(best_model_path, output_dir, train_loader, val_loader, transformer):
    cmbnd_model = TrajectoryModel(hidden_channels=256, num_layers=6)
    
    # Load the pre-trained model if it exists
    if best_model_path is not None:
        print(f"Loading pre-trained model from {best_model_path}")
        cmbnd_model.load(best_model_path)
    
    # Initialize optimizer with a lower learning rate for fine-tuning
    optimizer = torch.optim.AdamW(
        cmbnd_model.velocity_model.parameters(),
        lr=5e-6,  # Lower learning rate for fine-tuning
        weight_decay=1e-4
    )
    
    # Add learning rate scheduler with more conservative settings
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,  # More gradual learning rate reduction
        patience=8,   # More patience before reducing learning rate
        verbose=True,
        min_lr=1e-7
    )
    
    trn_criterion = WeightedMSELoss()
    val_criterion = MAPELoss()
    
    cmbnd_model.train()
    best_val_loss = torch.inf
    train_losses = []
    val_losses = []
    no_improvement_count = 0
    max_no_improvement = 15  # Increased patience for early stopping
    
    for epoch in range(100):  # Reduced number of epochs since we're fine-tuning
        total_loss = 0.0
        cmbnd_model.train()
        
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device=DEVICE, dtype=torch.float32)
            input_data = data[:, :15]
            target_velocity = data[:, 15:18]
            
            # Transform data
            graph_data = transformer.transform(input_data)
            positions = (input_data[:, :3], input_data[:, 12:15], input_data[:, 6:9])
            
            # Forward pass
            optimizer.zero_grad()
            pred_velocity = cmbnd_model(graph_data)
            velocity_loss = trn_criterion(pred_velocity, target_velocity, positions)
            
            # Backward pass with gradient clipping
            velocity_loss.backward()
            torch.nn.utils.clip_grad_norm_(cmbnd_model.velocity_model.parameters(), max_norm=0.5)  # Reduced gradient clipping
            optimizer.step()
            
            total_loss += velocity_loss.item()
            
            # Print detailed progress
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(train_loader)}, '
                        f'Loss: {velocity_loss.item():.4f}')
            
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        val_loss = validate(cmbnd_model, val_loader, val_criterion, transformer)
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)

        # Save best model and check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0
            cmbnd_model.save(os.path.join(output_dir, "best_vel_model.pth"))
            print(f"Saved new best model! Validation Loss: {val_loss:.4f}")
        else:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Plot losses every 5 epochs
        if (epoch + 1) % 5 == 0:
            plot_losses(train_losses, [], val_losses, [],
                       os.path.join(output_dir, f"losses_epoch_{epoch + 1}.png"))
            
            # Print sample predictions
            with torch.no_grad():
                sample_data = next(iter(val_loader))
                sample_data = sample_data.to(device=DEVICE, dtype=torch.float32)
                sample_input = sample_data[:, :15]
                sample_target = sample_data[:, 15:18]
                sample_graph = transformer.transform(sample_input)
                sample_pred = cmbnd_model(sample_graph)
                
                print("\nSample Predictions:")
                for i in range(min(3, len(sample_pred))):
                    print(f"\nSample {i}:")
                    print(f"Predicted Velocity: {sample_pred[i].cpu().numpy()}")
                    print(f"True Velocity:      {sample_target[i].cpu().numpy()}")
                    print(f"Start Pos: {sample_input[i][:3].cpu().numpy()}")
                    print(f"Gate Pos: {sample_input[i][12:15].cpu().numpy()}")
                    print(f"End Pos: {sample_input[i][6:9].cpu().numpy()}")
                    print(f"MAPE: {val_criterion(sample_pred[i:i+1], sample_target[i:i+1]).item():.2f}%")
                print("-" * 50)
    
    # Final evaluation
    final_val_loss = validate(cmbnd_model, val_loader, val_criterion, transformer)
    print(f"\nFinal Validation Loss: {final_val_loss:.4f}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    
    # Save final loss plots
    plot_losses(train_losses, [], val_losses, [],
               os.path.join(output_dir, "final_losses.png"))
    
    # Save loss history
    np.save(os.path.join(output_dir, "train_losses.npy"), np.array(train_losses))
    np.save(os.path.join(output_dir, "val_losses.npy"), np.array(val_losses))

if __name__ == "__main__":
    main() 
