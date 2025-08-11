import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
class MigrationAttention(nn.Module):
    """
    A simplified attention-based model that's easier to train and interpret.
    Focuses on direct state-pair relationships.
    """
    
    def __init__(self,n_states = 50,n_node_features = 11,n_edge_features = 5,hidden_dim = 128,n_heads= 4,dropout= 0.2):
        super().__init__()
        
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        
        # State and feature encoders
        self.state_encoder = nn.Sequential(
            nn.Linear(n_node_features, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.GELU()
        )
        
        # Edge feature encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(n_edge_features, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.GELU()
        )
        
        # Attention mechanism for state pairs
        self.attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        # Flow prediction network
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self,node_features,edge_features,edge_index,return_attention= False):
        """Simplified forward pass focusing on essential components."""
        
        if node_features.dim() == 2:
        # Unbatched: add batch dimension
            node_features = node_features.unsqueeze(0)
            edge_features = edge_features.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        # Encode state features
        state_embeds = self.state_encoder(node_features)  # [batch, n_states, hidden]
        
        # Get origin and destination embeddings
        row, col = edge_index
        origin_embeds = state_embeds[:, row, :]  # [batch, n_edges, hidden]
        dest_embeds = state_embeds[:, col, :]    # [batch, n_edges, hidden]
        
        # Apply attention: destination queries, origin provides keys/values
        attn_out, attn_weights = self.attention(
            dest_embeds, origin_embeds, origin_embeds
        )
        
        # Encode edge features
        edge_embeds = self.edge_encoder(edge_features)
        
        # Combine all information
        combined = torch.cat([origin_embeds, attn_out, edge_embeds], dim=-1)
        
        # Predict flows
        predictions = self.predictor(combined).squeeze(-1)
        
        if return_attention:
            return predictions, attn_weights.detach().cpu().numpy()
        return predictions, None
def train_transformer(model,train_data,val_data,epochs,lr= 0.001,device = 'cuda',patience= 200,gradient_clip=1.0):
    """
    Train the transformer model with validation-based early stopping.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2
    )
    
    criterion = nn.MSELoss()  # or use PoissonDevianceLoss() for count data
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_data:

            node_feat, edge_feat, targets, edge_idx = batch.x,batch.edge_features,batch.y,batch.edge_index_ALL
            
            # Move to device
            node_feat = node_feat.to(device)
            edge_feat = edge_feat.to(device)
            targets = targets.to(device)
            edge_idx = edge_idx.to(device)
            
            optimizer.zero_grad()
            predictions, _ = model(node_feat, edge_feat, edge_idx)
            
            # Ensure predictions and targets have same shape
            if predictions.dim() > targets.dim():
                predictions = predictions.squeeze()
            
            loss = criterion(predictions, targets)
            loss.backward()
            
            # Gradient clipping
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_data:
                node_feat, edge_feat, targets, edge_idx = batch.x,batch.edge_features,batch.y,batch.edge_index_ALL
                
                node_feat = node_feat.to(device)
                edge_feat = edge_feat.to(device)
                targets = targets.to(device)
                edge_idx = edge_idx.to(device)
                
                predictions, _ = model(node_feat, edge_feat, edge_idx)
                
                if predictions.dim() > targets.dim():
                    predictions = predictions.squeeze()
                
                loss = criterion(predictions, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_data)
        avg_val_loss = val_loss / len(val_data)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                model.load_state_dict(best_model_state)
                break
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    return model, train_losses, val_losses

def evaluate_transformer(model,test_data,device = 'cuda',return_predictions = False):
    """
    Evaluate transformer model performance.
    """
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_data:
            node_feat, edge_feat, targets, edge_idx = batch.x,batch.edge_features,batch.y,batch.edge_index_ALL
            
            node_feat = node_feat.to(device)
            edge_feat = edge_feat.to(device)
            targets = targets.to(device)
            edge_idx = edge_idx.to(device)
            
            predictions, _ = model(node_feat, edge_feat, edge_idx)
            
            if predictions.dim() > targets.dim():
                predictions = predictions.squeeze()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Additional metrics
    mae = np.mean(np.abs(all_targets - all_predictions))
    mape = np.mean(np.abs((all_targets - all_predictions) / (all_targets + 1e-8))) * 100
    
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MAPE: {mape:.2f}%")
    
    if return_predictions:
        return all_predictions, all_targets, {'mse': mse, 'r2': r2, 'mae': mae, 'mape': mape}
    
    return {'mse': mse, 'r2': r2, 'mae': mae, 'mape': mape}

def transformerSimple(n_node_features,n_edge_features,hidden_dim,n_heads,dropout,epochs,lr,train_data,val_data,test_data):
    model = MigrationAttention(
        n_states=50,
        n_node_features=n_node_features,  # economic indicators, demographics
        n_edge_features=n_edge_features,  # distance, historical flows
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        dropout=dropout
    )
    model, train_losses, val_losses = train_transformer(
        model, train_data, val_data,
        epochs=epochs,lr=lr
    )   
    predictions = evaluate_transformer(model,
        test_data
    )
    return model, train_losses, val_losses, predictions

def transformerComplex(n_node_features,n_edge_features,hidden_dim,n_heads,dropout,epochs,lr,train_data,val_data,test_data,layers,gradient_clip):
    model = MigrationAttention2(
        n_states=50,
        n_node_features=n_node_features,  # economic indicators, demographics
        n_edge_features=n_edge_features,  # distance, historical flows
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        dropout=dropout,
        n_layers=layers
    )
    model, train_losses, val_losses = train_transformer(
        model, train_data, val_data,
        epochs=epochs,lr=lr,gradient_clip=gradient_clip
    )   
    predictions = evaluate_transformer(model,
        test_data
    )
    return model, train_losses, val_losses, predictions

class MigrationAttention2(nn.Module):
    """
    Attention model operating on raw tensors (node_feat, edge_feat, edge_index).
    Compatible with current train/evaluate loops that call:
        predictions, _ = model(node_feat, edge_feat, edge_idx)
    """
    def __init__(self,n_states = 50,n_node_features = 11,n_edge_features = 5,
                 hidden_dim = 128,n_heads= 4,dropout= 0.2,n_layers= 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_states = n_states

        self.node_encoder = nn.Sequential(
            nn.Linear(n_node_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(n_edge_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.state_embedding = nn.Embedding(n_states, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pair_attention = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self,
                node_feat: torch.Tensor,
                edge_feat: torch.Tensor,
                edge_index: torch.Tensor,
                return_attention: bool = False):
        """
        Args:
            node_feat: [N, F_n]
            edge_feat: [E, F_e]
            edge_index: [2, E]
        Returns:
            (predictions, attn_weights_or_None)
        """
        if node_feat.dim() != 2:
            raise ValueError("Expected node_feat shape [N, F_n]")
        if edge_feat.dim() != 2:
            raise ValueError("Expected edge_feat shape [E, F_e]")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be shape [2, E]")

        N = node_feat.size(0)
        E = edge_index.size(1)

        node_embeds = self.node_encoder(node_feat)  # [N,H]
        if N == self.n_states:
            ids = torch.arange(N, device=node_feat.device)
            node_embeds = node_embeds + self.state_embedding(ids)

        # Global interaction among states
        node_embeds_tr = self.transformer(node_embeds.unsqueeze(0)).squeeze(0)  # [N,H]

        row, col = edge_index
        if row.max() >= N or col.max() >= N:
            raise ValueError("edge_index contains index >= number of nodes")

        origin_embeds = node_embeds_tr[row]  # [E,H]
        dest_embeds = node_embeds_tr[col]    # [E,H]

        # Pair attention across all edges (destination queries origin)
        origin_batch = origin_embeds.unsqueeze(0)  # [1,E,H]
        dest_batch = dest_embeds.unsqueeze(0)      # [1,E,H]
        attn_out, attn_weights = self.pair_attention(dest_batch, origin_batch, origin_batch)
        attn_out = attn_out.squeeze(0)  # [E,H]

        edge_embeds = self.edge_encoder(edge_feat)  # [E,H]

        combined = torch.cat([origin_embeds, attn_out, edge_embeds], dim=-1)  # [E,3H]
        predictions = self.edge_predictor(combined).squeeze(-1)  # [E]

        if return_attention:
            return predictions, attn_weights.detach()
        return predictions, None