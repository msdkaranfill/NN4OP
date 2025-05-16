import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
import numpy as np
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NormalizationLayer(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        
        # Initialize running statistics as parameters
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x):
        # Always use running statistics for TorchScript compatibility
        mean = self.running_mean.view(1, -1)
        var = self.running_var.view(1, -1)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Handle anomalies with a static operation
        x_norm = torch.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        
        return x_norm

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Main path with wider layers
        self.main_path = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.BatchNorm1d(channels * 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(channels * 2, channels),
            nn.BatchNorm1d(channels)
        )
        
        # Skip connection with learnable scaling
        self.register_buffer("skip_scale", torch.tensor(0.01))
        
    def forward(self, x):
        # Main path
        main_out = self.main_path(x)
        
        # Skip connection with learnable scaling
        skip_out = self.skip_scale * x
        
        # Combine with residual connection and ensure gradient flow
        out = main_out + skip_out
        return torch.tanh(out)  # Move tanh after addition for better gradient flow

class GraphDataTransformer:
    def __init__(self, device=DEVICE, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.eps = torch.tensor(1e-6, device=device, dtype=dtype)
        
    def transform(self, batch_data):
        # Ensure input is in correct dtype
        batch_data = batch_data.to(dtype=self.dtype)
        
        # Extract node features
        batch_size = batch_data.shape[0]
        start_pos = batch_data[:, :3]       # [batch_size, 3]
        start_vel = batch_data[:, 3:6]      # [batch_size, 3]
        end_pos = batch_data[:, 6:9]        # [batch_size, 3]
        end_vel = batch_data[:, 9:12]       # [batch_size, 3]
        gate_pos = batch_data[:, 12:15]     # [batch_size, 3]

        # Node features: [batch_size, 3 nodes, 6 features each] (position + velocity)
        # For gate node, we use zero velocity
        node_features = torch.stack([
            torch.cat([start_pos, start_vel], dim=1),  # Start node: [pos(3), vel(3)]
            torch.cat([gate_pos, torch.zeros_like(gate_pos)], dim=1),  # Gate node: [pos(3), vel(3)=0]
            torch.cat([end_pos, end_vel], dim=1)       # End node: [pos(3), vel(3)]
        ], dim=1)  # [batch_size, 3, 6]

        # Flatten node features for batched processing in PyG
        node_features = node_features.view(-1, 6)  # [batch_size * 3, 6]
        
        # Edge indices (same for all graphs): Start -> Gate, Gate -> End
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t().repeat(1, batch_size)  # [2, 2]
        
        # Adjust edge indices for batched graphs
        num_nodes = 3  # start, gate, end
        batch_offsets = torch.arange(0, batch_size * num_nodes, step=num_nodes,
                                   dtype=torch.long).repeat_interleave(2)
        batch_offsets = torch.stack([batch_offsets, batch_offsets], dim=0)
        edge_index += batch_offsets
        
        # Get gate indices for each graph in the batch
        gate_indices = torch.arange(1, batch_size * num_nodes, step=num_nodes)
        
        # Edge features (directions and relative velocities): [batch_size*2, 6]
        # For start->gate: direction and relative velocity (gate_vel - start_vel)
        # For gate->end: direction and relative velocity (end_vel - gate_vel)
        edge_features = torch.cat([
            torch.cat([gate_pos-start_pos, -start_vel], dim=1),  # Start -> Gate
            torch.cat([end_pos-gate_pos, end_vel], dim=1)        # Gate -> End
        ], dim=1)  # [batch_size*2, 6]
        edge_features = edge_features.view(edge_index.size(1), -1)

        # Create PyG Data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            gate_indices=gate_indices
        )
        graph_data.num_nodes = int(node_features.size(0))

        graph_data = graph_data.to(self.device)
            
        return graph_data

class CustomEdgeConv(nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn
        self.register_buffer("skip_scale", torch.tensor(0.1))
        self.activation = ScaledTanh(scale=4.0)
        self.direction_scale = ScaledTanh(scale=2.3094)
        
    def forward(self, x, edge_index, edge_attr):
        # Simple message passing
        row = edge_index[0]
        col = edge_index[1]
        x_i = x[row]
        x_j = x[col]
        
        # Basic operations only
        rel_pos = x_j - x_i
        direction = self.direction_scale(rel_pos)
        message_input = torch.cat([x_i, x_j, edge_attr, direction], dim=-1)
        messages = self.nn(message_input)
        
        # Vectorized aggregation
        out = torch.zeros_like(x)
        out.index_reduce_(0, col, messages, reduce='amax')
        
        # Residual
        out = out + self.skip_scale * x
        return self.activation(out)

class WeightedMSELoss(nn.Module):
    def __init__(self, device=DEVICE, dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.register_buffer('eps', torch.tensor(1e-3, device=device, dtype=dtype))
        
        # Base scaling factors with minimum scale of 10.0
        self.dir_scale = 15.0 
        self.mag_scale = 15.0  
        
    def smooth_abs(self, x, alpha=0.01):
        """Smooth approximation of absolute value for better gradient flow"""
        return torch.sqrt(x * x + alpha)
        
    def forward(self, pred_velocity, true_velocity, positions):
        # Calculate norms once and reuse
        pred_norm = torch.norm(pred_velocity, dim=1)
        true_norm = torch.norm(true_velocity, dim=1)
        
        # Calculate direction similarity using cosine similarity
        dir_similarity = torch.sum(pred_velocity * true_velocity, dim=1) / (pred_norm * true_norm + self.eps)
        
        # Get sample-wise maximum velocity magnitude using smooth_abs
        true_max = torch.max(self.smooth_abs(true_velocity), dim=1)[0]
        
        # Handle zero velocity cases
        zero_vel_mask = true_norm < self.eps
        
        # Calculate magnitude ratios for all cases using masking
        mag_ratio = torch.where(
            zero_vel_mask,
            pred_norm,  # For zero velocities, use pred_norm directly
            (pred_norm / (true_norm + self.eps)) * true_max  # For non-zero velocities
        )
        
        # Calculate magnitude difference using component-wise errors
        # Scale errors by max velocity and sum them up
        component_errors = self.smooth_abs(pred_velocity - true_velocity) * 2.3094
        mag_diff = torch.sum(component_errors, dim=1)  # Sum errors across components
        
        # Calculate MSE
        mse = F.mse_loss(pred_velocity, true_velocity, reduction='none').mean(dim=1)
        
        # Calculate base losses
        dir_loss = (1.0 - dir_similarity) * self.dir_scale
        mag_loss = mag_diff * self.mag_scale
        
        # Dynamic MSE scaling based on directional error
        mse_scale = 1.0 + (1.0 - dir_similarity)  # Increase MSE weight when direction is off
        mse_loss = mse * mse_scale * 5
        
        # Calculate dynamic weights based on relative loss contributions
        # Normalize losses to [0,1] range first
        max_dir_loss = dir_loss.max()
        max_mag_loss = mag_loss.max()
        norm_dir_loss = dir_loss / (max_dir_loss + self.eps)
        norm_mag_loss = mag_loss / (max_mag_loss + self.eps)
        
        # Calculate weights using normalized losses
        # Higher normalized loss -> higher weight
        dir_weight = norm_dir_loss
        mag_weight = norm_mag_loss
        
        # Normalize weights to sum to 1
        weight_sum = dir_weight + mag_weight
        dir_weight = dir_weight / weight_sum
        mag_weight = mag_weight / weight_sum
        
        # Combine losses with adaptive weights and add MSE
        total_loss = (
            dir_weight * dir_loss +
            mag_weight * mag_loss +
            mse_loss
        )
        
        # Print batch statistics periodically
        if torch.rand(1) < 0.01:  # 1% chance to print
                    with torch.no_grad():
                        print("\n=== BATCH STATISTICS ===\n")
                        print(f"Loss Components - Dir: {dir_loss.mean():.3f}, Mag: {mag_loss.mean():.3f}, MSE: {mse_loss.mean():.3f}")
                        print(f"Average Weights - Dir: {dir_weight.mean():.3f}, Mag: {mag_weight.mean():.3f}")
                        print(f"Total Loss: {total_loss.mean():.3f}")
                        print("-" * 50)
                           
        # Print random sample for detailed analysis
        if torch.rand(1) < 0.01:  # 1% chance to print
            with torch.no_grad():
                idx = torch.randint(0, len(pred_velocity), (1,)).item()
                print("\n=== RANDOM SAMPLE ANALYSIS ===\n")
                print(f"Sample {idx}:")
                print(f"Pred Velocity: {pred_velocity[idx].cpu().numpy()}")
                print(f"True Velocity: {true_velocity[idx].cpu().numpy()}")
                print(f"Direction Similarity: {dir_similarity[idx]:.4f}")
                print(f"Magnitude Ratio: {mag_ratio[idx]:.4f}")
                print(f"Component Errors: {component_errors[idx].cpu().numpy()}")
                print(f"Magnitude Diff: {mag_diff[idx]:.4f}")
                print(f"MSE: {mse[idx]:.4f}")
                print(f"Base Losses - Dir: {dir_loss[idx]:.4f}, Mag: {mag_loss[idx]:.4f}, MSE: {mse_loss[idx]:.4f}")
                print(f"Weights - Dir: {dir_weight[idx]:.4f}, Mag: {mag_weight[idx]:.4f}")
                print(f"Total Loss: {total_loss[idx]:.4f}")
                print("-" * 50)
        
        # Print worst predictions with more detailed analysis
        if total_loss.mean() > 100:  # Changed from sum() to mean()
            with torch.no_grad():
                # Get indices of top 5 worst predictions
                worst_indices = torch.topk(total_loss, k=min(5, len(total_loss)))[1]
                
                print("\n=== HIGH LOSS ANALYSIS ===\n")
                for i, idx in enumerate(worst_indices):
                    print(f"Sample {i}:")
                    print(f"Pred Velocity: {pred_velocity[idx].cpu().numpy()}")
                    print(f"True Velocity: {true_velocity[idx].cpu().numpy()}")
                    print(f"Direction Similarity: {dir_similarity[idx]:.4f}")
                    print(f"Magnitude Ratio: {mag_ratio[idx]:.4f}")
                    print(f"Component Errors: {component_errors[idx].cpu().numpy()}")
                    print(f"Magnitude Diff: {mag_diff[idx]:.4f}")
                    print(f"MSE: {mse[idx]:.4f}")
                    print(f"Base Losses - Dir: {dir_loss[idx]:.4f}, Mag: {mag_loss[idx]:.4f}, MSE: {mse_loss[idx]:.4f}")
                    print(f"Weights - Dir: {dir_weight[idx]:.4f}, Mag: {mag_weight[idx]:.4f}")
                    print(f"Total Loss: {total_loss[idx]:.4f}")
                    print("-" * 50)
        
        return total_loss.mean()

class MAPELoss(nn.Module):
    #only for validation, bcs it can cause exploding gradients
    def __init__(self, eps=1e-6, device=DEVICE, dtype=torch.float32):
        super().__init__()
        self.eps = torch.tensor(eps, dtype=dtype, device=device)
        self.device = device
        self.dtype = dtype
        self.vel_weight = 0.6
        self.time_weight = 0.4

    def forward(self, pred_velocity, true_velocity, pred_time=None, true_time=None):
        # Move to correct device and dtype
        pred_velocity = pred_velocity.to(device=self.device, dtype=self.dtype)
        true_velocity = true_velocity.to(device=self.device, dtype=self.dtype)
        # Compute velocity magnitude MAPE
        pred_vel_mag = torch.norm(pred_velocity, dim=1) + self.eps
        true_vel_mag = torch.norm(true_velocity, dim=1) + self.eps
        valid_mask = true_vel_mag > 0.01  # to keep from blow-up
        safe_true_vel = true_vel_mag[valid_mask]
        safe_pred_vel = pred_vel_mag[valid_mask]
        if safe_true_vel.numel() > 0:
            vel_mape = torch.mean(torch.abs((safe_pred_vel - safe_true_vel) / safe_true_vel)) * 100.0
        else:
            # Log fallback warning with example values
            print("\n[Warning] All velocity magnitudes were near-zero! Falling back to batch-level MAPE estimate.")

            # Calculate full batch velocity MAPE (ignoring masking)
            full_pred_mag = torch.norm(pred_velocity, dim=1) + self.eps
            full_true_mag = torch.norm(true_velocity, dim=1) + self.eps
            batch_vel_mape = torch.mean(torch.abs((full_pred_mag - full_true_mag) / full_true_mag)) * 100.0
            vel_mape = batch_vel_mape

        if pred_time is not None and true_time is not None:
            pred_time = pred_time.to(device=self.device, dtype=self.dtype)
            true_time = true_time.to(device=self.device, dtype=self.dtype)
            true_time = true_time.view_as(pred_time)
            time_mape = torch.mean(torch.abs((pred_time - true_time) / (true_time + self.eps))) * 100.0
            vel_loss = self.vel_weight * vel_mape 
            time_loss = self.time_weight * time_mape

            # Print a few values occasionally
            if torch.rand(1) < 0.01:
                with torch.no_grad():
                    print("\nMAPE Monitoring:")
                    print(f"Velocity MAPE: {vel_mape.item()}")
                    if time_mape is not None:
                        print(f"Time MAPE:     {time_mape.item()}")
                        print("-" * 50)

            return vel_loss, time_loss
        
        vel_loss = vel_mape 
        return vel_loss

"""
class ScaledTanhFunction4Training(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.save_for_backward(x, scale)
        return scale * torch.tanh(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, scale = ctx.saved_tensors
        # Compute gradient of tanh(x) with a minimum gradient to prevent vanishing
        tanh_grad = 1 - torch.tanh(x) ** 2
        # Add a small constant to prevent gradient from going to zero
        tanh_grad = torch.clamp(tanh_grad, min=0.1)
        # Chain rule: dL/dx = dL/dy * dy/dx
        return grad_output * scale * tanh_grad, grad_output * torch.tanh(x)"""
    

class ScaledTanh(nn.Module):
    def __init__(self, scale=4.7094):
        super().__init__()
        # Use a constant tensor instead of a parameter
        self.register_buffer("scale", torch.tensor(scale))
        
    def forward(self, x):
        return self.scale * torch.tanh(x)


class VelocityTransformer(nn.Module):
    def __init__(self, hidden_channels=256, num_heads=8, num_layers=2):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Feed-forward network with ScaledTanh
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            ScaledTanh(scale=4.0),
            nn.Dropout(0.1),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels)
        )
        
        # Output projection with gradual dimension reduction and ScaledTanh
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            ScaledTanh(scale=4.0),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.LayerNorm(hidden_channels // 2),
            ScaledTanh(scale=4.0),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.LayerNorm(hidden_channels // 4),
            ScaledTanh(scale=4.0),
            nn.Linear(hidden_channels // 4, 3),
            ScaledTanh(scale=2.3094)  # Final scale matches max velocity
        )
        
    def forward(self, x: torch.Tensor, gate_indices: torch.Tensor) -> torch.Tensor:
        # Extract gate node features with fixed shape [batch_size, 1, hidden_channels]
        batch_size = gate_indices.size(0)
        # Create tensor with explicit device and dtype
        gate_features = torch.zeros(batch_size, 1, self.hidden_channels, device=x.device, dtype=x.dtype)
        # Get gate features and ensure proper shape
        gate_features = x[gate_indices].unsqueeze(1)  # Shape: [batch_size, 1, hidden_channels]
        
        # Self-attention with residual
        attn_output, _ = self.self_attn(gate_features, gate_features, gate_features)
        attn_output = gate_features + attn_output
        
        # Feed-forward with residual
        ffn_output = self.ffn(attn_output)
        ffn_output = attn_output + ffn_output
        
        # Project to velocity with gradual dimension reduction
        # Output shape: [batch_size, 1, 3]
        velocity = self.output_proj(ffn_output)
        
        # Reshape to [batch_size, 3] with fixed shape
        return velocity.reshape(batch_size, 3)

class TimeLoss(nn.Module):
    def __init__(self, device=DEVICE, dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.eps = torch.tensor(1e-6, device=device, dtype=dtype)
        
    def forward(self, pred_time, true_time):
        # Calculate absolute error
        abs_error = torch.abs(pred_time - true_time)
        
        # Calculate relative error with better gradient properties
        # Using log1p to handle small differences better
        rel_error = torch.log1p(abs_error / (true_time + self.eps))
        
        # Combine errors with exponential penalty for larger deviations
        # This makes the loss more sensitive to larger errors while maintaining good gradients
        combined_error = rel_error * torch.exp(abs_error / 10.0)  # Increased scale factor to 10.0 for smoother gradient
        
        # Final loss is just the combined error
        loss = combined_error.mean()
        
        return loss

class TimeModel(nn.Module):
    def __init__(self, hidden_channels=256, num_layers=2):
        super().__init__()
        
        # Input features: [x, y, z, vx, vy, vz] for each node
        self.input_size = 6  # 6 features per node
        
        # Combined feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(self.input_size * 3, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.Tanh(),
            nn.Dropout(0.2)
        )
        
        # Multiple hidden layers with residual connections
        hidden_layers = []
        for _ in range(num_layers):
            # Create layers
            linear = nn.Linear(hidden_channels, hidden_channels)
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='tanh')
            nn.init.constant_(linear.bias, 0.1)
            
            layer_norm = nn.LayerNorm(hidden_channels)
            nn.init.ones_(layer_norm.weight)
            nn.init.zeros_(layer_norm.bias)
            
            hidden_layers.append(nn.Sequential(
                linear,
                layer_norm,
                nn.Tanh(),
                nn.Dropout(0.2)
            ))
        self.hidden_layers = nn.ModuleList(hidden_layers)
        
        # Final layers with more gradual dimension reduction
        final_layers = []
        
        # First reduction
        linear1 = nn.Linear(hidden_channels, hidden_channels // 2)
        nn.init.kaiming_normal_(linear1.weight, mode='fan_in', nonlinearity='tanh')
        nn.init.constant_(linear1.bias, 0.1)
        layer_norm1 = nn.LayerNorm(hidden_channels // 2)
        nn.init.ones_(layer_norm1.weight)
        nn.init.zeros_(layer_norm1.bias)
        final_layers.extend([linear1, layer_norm1, nn.Tanh(), nn.Dropout(0.1)])
        
        # Second reduction
        linear2 = nn.Linear(hidden_channels // 2, hidden_channels // 4)
        nn.init.kaiming_normal_(linear2.weight, mode='fan_in', nonlinearity='tanh')
        nn.init.constant_(linear2.bias, 0.1)
        final_layers.extend([linear2, nn.Tanh()])
        
        # Third reduction
        linear3 = nn.Linear(hidden_channels // 4, hidden_channels // 8)
        nn.init.kaiming_normal_(linear3.weight, mode='fan_in', nonlinearity='tanh')
        nn.init.constant_(linear3.bias, 0.1)
        final_layers.extend([linear3, nn.Tanh()])
        
        # Final layer
        linear4 = nn.Linear(hidden_channels // 8, 1)
        nn.init.kaiming_normal_(linear4.weight, mode='fan_in', nonlinearity='tanh')
        nn.init.constant_(linear4.bias, 0.1)
        final_layers.extend([linear4, nn.Softplus()])
        
        self.final_layers = nn.Sequential(*final_layers)
        
    def forward(self, x, pred_velocity):
        # Get batch size and reshape if needed
        batch_size = x.size(0) // 3  # 3 nodes per sample
        num_nodes = 3  # start, gate, end nodes
        
        # Reshape to [batch_size, num_nodes, features]
        x = x.view(batch_size, num_nodes, -1)
        
    
        x = x.clone()  # Create a copy to avoid in-place operations
        x[:, 1, 3:6] = pred_velocity
    
        # Flatten all node features
        x = x.view(batch_size, -1)  # [batch_size, num_nodes * features]
        
        # Process all features together
        out = self.feature_processor(x)
        
        # Process through hidden layers with residual connections
        for layer in self.hidden_layers:
            residual = out
            out = layer(out)
            out = out + residual  # Residual connection
        
        # Final processing with gradual dimension reduction
        out = self.final_layers(out)
        
        return out

class VelocityModel(nn.Module):
    def __init__(self, hidden_channels=256, num_layers=6):
        super().__init__()
        
        # Store size parameters as Python integers
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # Initialize layers
        self.node_encoder = nn.Linear(6, hidden_channels)  # 6 features per node (3 pos + 3 vel)
        self.edge_encoder = nn.Linear(6, hidden_channels)  # 6 features per edge (3 dir + 3 rel vel)
        
        # Graph convolution layers
        self.convs = nn.ModuleList([
            CustomEdgeConv(
                nn=nn.Sequential(
                    nn.Linear(4 * hidden_channels, hidden_channels * 2),
                    nn.Tanh(),
                    nn.Linear(hidden_channels * 2, hidden_channels)),
            ) for _ in range(num_layers)
        ])
        
        # Velocity transformer
        self.velocity_transformer = VelocityTransformer(
            hidden_channels=hidden_channels,
            num_heads=8,
            num_layers=2
        )
        
    def forward(self, x, edge_index, edge_attr, gate_indices):
        # Encode node and edge features
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Apply graph convolutions
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        
        # Get velocity predictions using transformer
        pred_velocity = self.velocity_transformer(x, gate_indices)
        
        return pred_velocity

class TrajectoryModel(nn.Module):
    def __init__(self, hidden_channels=256, num_layers=6):
        super().__init__()
        
        # Initialize both models
        self.velocity_model = VelocityModel(
            hidden_channels=hidden_channels,
            num_layers=num_layers
        )
        
        self.time_model = TimeModel(
            hidden_channels=hidden_channels,
            num_layers=num_layers
        )
        
        # Edge feature processor for component attention
        self.edge_processor = nn.Sequential(
            nn.Linear(12, hidden_channels // 2),  # Process concatenated edge features
            nn.LayerNorm(hidden_channels // 2),
            nn.Tanh(),
            nn.Linear(hidden_channels // 2, 3),  # 3 components
            nn.Softmax(dim=-1)
        )
        
        # Move entire model to device
        self.to(device=DEVICE, dtype=torch.float32)
        
    def forward(self, x, edge_index, edge_attr, gate_indices):
        batch_size = x.size(0) // 3  # 3 nodes per sample
        
        # Reshape edge_index to separate graphs
        edge_index = edge_index.view(2, batch_size, 2)  # [2, batch_size, num_edges_per_graph]
        
        # Find edges connected to gate nodes
        # incoming: target == gate, outgoing: source == gate
        incoming_edges = (edge_index[1] == gate_indices.unsqueeze(1))  # [batch_size, num_edges_per_graph]
        outgoing_edges = (edge_index[0] == gate_indices.unsqueeze(1))  # [batch_size, num_edges_per_graph]
        
        # Combine masks
        relevant_edges = incoming_edges | outgoing_edges  # [batch_size, num_edges_per_graph]
        
        # Reshape edge features to match batch structure
        edge_features = edge_attr.view(batch_size, 2, 6)  # [batch_size, num_edges_per_graph, 6]
        
        # Mask out irrelevant edges and flatten
        edge_features = edge_features * relevant_edges.unsqueeze(-1)  # [batch_size, num_edges_per_graph, 6]
        edge_features = edge_features.view(batch_size, -1)  # [batch_size, 12]
        
        # Calculate component weights
        component_weights = self.edge_processor(edge_features)  # [batch_size, 3]
        
        # Create zero tensor and set gate node's velocity columns
        first_outs = torch.zeros_like(x)
        first_outs[gate_indices, 3:] = component_weights
        
        # Add component weights to node features
        x = x + first_outs
        
        # Get velocity predictions using velocity model
        pred_velocity = self.velocity_model(x, edge_index.view(2, -1), edge_attr, gate_indices)
        
        # Process through time model
        pred_time = self.time_model(x, pred_velocity)
        
        return pred_velocity, pred_time
        
    def save(self, path):
        """Save model state dict"""
        state_dict = {
            'velocity_model': self.velocity_model.state_dict(),
            'time_model': self.time_model.state_dict(),
            'edge_processor': self.edge_processor.state_dict()
        }
        # Ensure all tensors are on CPU before saving
        state_dict = {k: v.cpu() for k, v in state_dict.items()}
        torch.save(state_dict, path)
        
    def load(self, path):
        """Load model state dict"""
        try:
            #print(f"\nAttempting to load model from: {path}")
            if not os.path.exists(path):
                print(f"ERROR: File does not exist at path: {path}")
                return False
                
            checkpoint = torch.load(path)
            #print(f"Successfully loaded checkpoint from {path}")
            
            if isinstance(checkpoint, dict):
                #print("Checkpoint is a dictionary")
                #print("Available keys:", checkpoint.keys())
                
                if 'velocity_model' in checkpoint:
                    #print("\nLoading velocity model from checkpoint['velocity_model']")
                    #print("Velocity model state dict keys:", checkpoint['velocity_model'].keys())
                    self.velocity_model.load_state_dict(checkpoint['velocity_model'])
                    #print("Velocity model loaded successfully")
                    # Print some weight statistics
                    #for name, param in self.velocity_model.named_parameters():
                        #if 'weight' in name:
                        #print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
                elif 'model_state_dict' in checkpoint:
                    #print("\nLoading velocity model from checkpoint['model_state_dict']")
                    self.velocity_model.load_state_dict(checkpoint['model_state_dict'])
                    #print("Velocity model loaded successfully")
                else:
                    print("ERROR: No valid velocity model state dict found in checkpoint")
                    return False
                    
                if 'time_model' in checkpoint and checkpoint['time_model'] is not None:
                    self.time_model.load_state_dict(checkpoint['time_model']) 
                if 'edge_processor' in checkpoint:
                    self.edge_processor.load_state_dict(checkpoint['edge_processor'])
            else:
                #print("\nCheckpoint is not a dictionary, attempting to load as direct state dict")
                #print("State dict keys:", checkpoint.keys())
                self.velocity_model.load_state_dict(checkpoint)
                #print("Velocity model loaded successfully")
                # Print some weight statistics
                #for name, param in self.velocity_model.named_parameters():
                    #if 'weight' in name:
                        #print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
            
            return True
            
        except Exception as e:
            print(f"ERROR loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
       
