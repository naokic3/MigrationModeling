import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
class BaselineMLP(nn.Module):
    # The num_edge_features argument is removed
    def __init__(self, num_node_features, hidden_dim=128):
        super(BaselineMLP, self).__init__()
        
        # MODIFIED: Input size is now just the two concatenated node feature vectors
        input_dim = num_node_features * 2
        
        self.edge_predictor = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_dim, hidden_dim // 2),
    nn.LeakyReLU(),
    nn.Dropout(0.2),
    # MODIFIED: The final layer outputs a single value for edge prediction
    nn.Linear(hidden_dim //2, 1)
)
        
    # MODIFIED: The edge_attr argument is removed
    def forward(self, x, edge_index):
        row, col = edge_index
        
        source_node_features = x[row]
        dest_node_features = x[col]
        
        # MODIFIED: Concatenate only the node features
        combined_features = torch.cat([
            source_node_features, 
            dest_node_features
        ], dim=1)
        
        predictions = self.edge_predictor(combined_features).squeeze(-1)
        
        return predictions
    
def train_modelMLP(model, train_loader, val_loader, device, epochs=100, lr=0.01):
    # The model should be moved to the device before creating the optimizer
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    # 1. Initialize lists to store loss history
    train_losses = []
    val_losses = []

    # --- Main Training Loop ---
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0 # 2. Initialize train_loss for the current epoch
        for data in train_loader:
            # Move the current batch of data to the device
            data = data.to(device)
            
            optimizer.zero_grad()
            predictions = model(data.x, data.edge_index_ALL)
            loss = loss_fn(predictions, data.y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 3. CORRECTED INDENTATION: This block now runs once per epoch
        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # 4. Iterate over the correct val_loader
            for data in val_loader:
                # 5. Move validation data to the device
                data = data.to(device)
                predictions = model(data.x, data.edge_index_ALL)
                # Use the consistent loss_fn
                loss = loss_fn(predictions, data.y)
                val_loss += loss.item()
                
        # 6. Calculate average losses correctly
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
                
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
                
        if (epoch + 1) % 10 == 0: # Log every 10 epochs
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses
def evaluate_modelMLP(model, test_data, device, scaler=None,scale=None):
    scaler = scale
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data in test_data:
            data = data.to(device)
            predictions = model(data.x, data.edge_index_ALL)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(data.y.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # Calculate metrics on normalized values
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    mae = mean_absolute_error(all_targets, all_predictions)
    
    # Calculate MAPE robustly by handling potential division by zero
    mask = all_targets != 0
    mape = np.mean(np.abs((all_targets[mask] - all_predictions[mask]) / all_targets[mask])) * 100

    
    predictions_denorm = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
    targets_denorm = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()

    predictions_original = np.exp(predictions_denorm) - 1
    targets_original = np.exp(targets_denorm) - 1

        # Calculate metrics on original scale
    mse_original = mean_squared_error(targets_original, predictions_original)
    mae_original = mean_absolute_error(targets_original, predictions_original)
        
        # Calculate MAPE for original scale
    mask_original = targets_original != 0
    mape_original = np.mean(np.abs((targets_original[mask_original] - predictions_original[mask_original]) / targets_original[mask_original])) * 100

    print(f"Normalized MSE: {mse:.4f}")
    print(f"Normalized R²: {r2:.4f}")
    print(f"Normalized MAE: {mae:.4f}")
    print(f"Normalized MAPE: {mape:.2f}%")
    print("-------------------------------")
    print(f"Original Scale MSE: {mse_original:.2f}")
    print(f"Original Scale MAE: {mae_original:.2f}")
    print(f"Original Scale MAPE: {mape_original:.2f}%")
    

    return all_predictions, all_targets








# SIMPLE GNN MODEL
class SimpleGNN(nn.Module):
    def __init__(self, num_node_features,num_edge_features,hidden_dim_G=128,hidden_dim_N=256,num_layers=1,heads=4):
        super(SimpleGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(num_node_features, hidden_dim_G,heads=heads, edge_dim=num_edge_features))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim_G, hidden_dim_G,heads=heads,edge_dim=num_edge_features))

        self.convs.append(GATConv(hidden_dim_G * heads, hidden_dim_G,heads=1, edge_dim=num_edge_features))
        
        self.edge_predictor = nn.Sequential(
    nn.Linear(hidden_dim_G*2, hidden_dim_N),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_dim_N, hidden_dim_N // 2),
    nn.LeakyReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_dim_N//2, 1)
)
    def forward(self, data):
        # 1. Generate node embeddings using GCN layers
        x, edge_index_GNN,edge_attr_GNN,edge_index_ALL = data.x, data.edge_index_GNN,data.edge_features, data.edge_index_ALL
        for i, conv in enumerate(self.convs):
            
            x = conv(x, edge_index_GNN,edge_attr=edge_attr_GNN)

            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        
        # 2. Predict edge values using the efficient vectorized path
        # Unpack the source and destination nodes for prediction
        node_embeddings = x
        row,col = edge_index_ALL
        start_nodes = node_embeddings[row]
        end_nodes = node_embeddings[col]
        
        edge_features = torch.cat([start_nodes,end_nodes], dim=1)
        print(edge_features.shape)
        predictions = self.edge_predictor(edge_features).squeeze()
        
        return predictions

def train_model1(model, train_data, val_data, epochs=100, lr=0.01, device='cpu'):
    # Move the model to the device (an alternative to doing it outside)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for snapshot in train_data:
            # THE KEY CHANGE: Move the data snapshot to the GPU
            snapshot = snapshot.to(device)
            
            optimizer.zero_grad()
            predictions = model(snapshot)
            loss = F.mse_loss(predictions, snapshot.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
    
        # --- Validation Loop ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for snapshot in val_data:
                # THE KEY CHANGE: Move validation data to the GPU
                snapshot = snapshot.to(device)
                
                predictions = model(snapshot)
                loss = F.mse_loss(predictions, snapshot.y)
                val_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_data)
        avg_val_loss = val_loss / len(val_data)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return train_losses, val_losses




def evaluate_model1(model, test_data, scaler=None):
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_data:
            predictions = model(data)
            
            # Store predictions and targets
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(data.y.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics on normalized values
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    # If scaler provided, denormalize for interpretable metrics
    if scaler:
        predictions_denorm = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
        targets_denorm = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
        
        # Convert from log space
        predictions_original = np.exp(predictions_denorm) - 1
        targets_original = np.exp(targets_denorm) - 1
        
        # Calculate metrics on original scale
        mse_original = mean_squared_error(targets_original, predictions_original)
        
        print(f"Normalized MSE: {mse:.4f}")
        print(f"Normalized R²: {r2:.4f}")
        print(f"Original Scale MSE: {mse_original:.2f}")
    else:
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
    
    return all_predictions, all_targets

def evaluate_model(model, test_data, scaler=None, device='cpu'):
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_data:
            # THE KEY CHANGE: Move the data snapshot to the GPU
            data = data.to(device)
            
            predictions = model(data)
            
            # Store predictions and targets
            # .cpu() moves data back to the CPU before converting to NumPy
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(data.y.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # --- The rest of the function is the same ---
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    if scaler:
        predictions_denorm = scaler.inverse_transform(all_predictions.reshape(-1, 1)).flatten()
        targets_denorm = scaler.inverse_transform(all_targets.reshape(-1, 1)).flatten()
        
        predictions_original = np.exp(predictions_denorm) - 1
        targets_original = np.exp(targets_denorm) - 1
        
        mse_original = mean_squared_error(targets_original, predictions_original)
        
        print(f"Normalized MSE: {mse:.4f}")
        print(f"Normalized R²: {r2:.4f}")
        print(f"Original Scale MSE: {mse_original:.2f}")
    else:
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")

    mae = np.mean(np.abs(all_targets - all_predictions))
    mape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + 1e-8))) * 100
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    return all_predictions, all_targets