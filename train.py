import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from trajectory_model import TrajectoryModel, WeightedMSELoss
from loadfiles import TrajectoryOptimizationDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def train_epoch(model, train_loader, criterion, optimizer, device, dtype=torch.float32):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc='Training'):
        # Get data
        inputs = batch[:, :15].to(device).to(dtype)
        velocity_targets = batch[:, 15:18].to(device).to(dtype)
        time_targets = batch[:, -1].reshape(-1, 1).to(device).to(dtype) 
        
        # Forward pass
        pred_velocity, pred_time = model(inputs)
        
        # Calculate loss
        loss = criterion(pred_velocity, velocity_targets, pred_time, time_targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device, dtype=torch.float32):
    model.eval()
    total_loss = 0
    s_count = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            # Get data
            inputs = batch[:, :15].to(device).to(dtype)
            velocity_targets = batch[:, 15:18].to(device).to(dtype)
            time_targets = batch[:, -1].reshape(-1, 1).to(device).to(dtype)
            
            # Forward pass
            pred_velocity, pred_time = model(inputs)
            
            # Calculate loss
            loss = criterion(pred_velocity, velocity_targets, pred_time, time_targets)
            total_loss += loss.item()
            
           
            s_count += batch.shape[0]
            if s_count % 500 == 0:
                for vo, vt, time_out, target2 in zip(pred_velocity.cpu(), velocity_targets.cpu(), 
                                                     pred_time.cpu(), time_targets.cpu()):
                    print("|".join(f"{oi:.4f} | {ti:.4f}" for oi, ti in zip(vo, vt)))
                    print("|"+(f"{time_out.item():.4f} | {target2.item():.4f}")) 

    return (total_loss / len(val_loader))

def plot_losses(train_losses, val_losses, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def main():
    # Parameters
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset = TrajectoryOptimizationDataset(os.path.join(BASE_DIR, "samples2"))
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True)
    
    # Initialize model
    model = TrajectoryModel(input_size=15, hidden_size=256, num_blocks=4).to(DEVICE)
    
    # Initialize criterion and optimizer
    criterion = WeightedMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    patience = 15
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, DEVICE)
        val_losses.append(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        #print(f"Velocity Error: {vel_error:.4f}")
        #print(f"Time Error: {time_error:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Plot losses
        plot_losses(train_losses, val_losses, 'loss_plot.png')
    
    # Final evaluation on test set
    model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
    test_loss = validate(model, test_loader, criterion, DEVICE)
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Velocity Error: {test_vel_error:.4f}")
    print(f"Test Time Error: {test_time_error:.4f}")

if __name__ == '__main__':
    main() 