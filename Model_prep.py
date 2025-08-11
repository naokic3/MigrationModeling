from networkx import nodes
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

def target_prep(target,pop_16,train_years):
    target = normalize_by_pop(target, pop_16)
    target_r = []
    for df in target:
        # Log transform to handle skewness
        df['result_rate'] = np.log(df['result_rate'] * 100000 + 1)
        target_r.append(df['result_rate'])
    target,scaler_target = standardize(target_r,train_years)
    target_tensors = [torch.tensor(year_data.flatten(), dtype=torch.float32) 
                  for year_data in target]
    return target_tensors, scaler_target

def normalize_by_pop(target, pop_16):
    norm_bypop = []
    for numerator_df, denominator_series in zip(target, pop_16):
        numerator_df['aligned_divisor'] = numerator_df['origin_state'].map(denominator_series)
        numerator_df['result_rate'] = numerator_df['flow'] / numerator_df['aligned_divisor']
        #numerator_df['result_rate'] = numerator_df['flow']
        norm_bypop.append(numerator_df)
    return norm_bypop

def standardize(target_log, train_years):

    train_idx = list(range(len(train_years)))
    #print(train_idx)
    training_data = np.concatenate([target_log[i] for i in train_idx])
    training_data_reshaped = training_data.reshape(-1, 1)
    scaler = StandardScaler()
    scaler.fit(training_data_reshaped)
    standardized = []
    for x in target_log:
        data = scaler.transform(x.values.reshape(-1, 1))
        standardized.append(data)
    
    return standardized, scaler

def node_normalization(nodes, train_years):
#FIT THE SCALER ON TRAINING DATA for node features
    train_combined_nodes = pd.concat([nodes[year] for year in train_years if year in nodes])
    scaler = StandardScaler()
    scaler.fit(train_combined_nodes)
    scaled_nodes = {}
    for year, df in nodes.items():
        scaled_nodes[year] = pd.DataFrame(
            scaler.transform(df), 
            index=df.index, 
            columns=df.columns
        )
    return scaled_nodes,scaler

import copy
def node_prep(nodes, train_years, YEARS):
    node,scaler_nodes = node_normalization(nodes, train_years)
    
    node_tensors = []
    for year in YEARS:
        node[year] = torch.from_numpy(node[year].to_numpy()).float()
        node_tensors.append(node[year])
    return node_tensors, scaler_nodes

def create_data_objects(node_features_list, edge_index_GNN,edge_index_ALL,edge_feat, target_tensors, years):
    """
    Create PyG Data objects for each year
    """
    data_list = []
    
    for i, year in enumerate(years):
        data = Data(
            x=node_features_list[i],
            edge_index_GNN=edge_index_GNN[i],
            edge_index_ALL=edge_index_ALL[0],
            y=target_tensors[i],
            year=year,
            edge_features=edge_feat[i]

        )
        data_list.append(data)
    
    return data_list

def edge_format(edges, nodes):
    target = []
    for year,df in edges.items():
        df = df.stack().reset_index(name='flow')
        df = df.rename(columns={'level_1': 'origin_state', 'NAME': 'destination_state'})
        df = df[df['origin_state'] != df['destination_state']]
        target.append(df)

    pop_16 = []
    for year,df in nodes.items():
        df = df['pop_16+']
        pop_16.append(df)

    return target, pop_16
def Knn_edges(edge_tensor,k=5):
    
    edge_index = []
    for df in edge_tensor:
        top_3_df = df.groupby('destination_state', group_keys=False) \
             .apply(lambda x: x.nlargest(k, 'result_rate'))
        edge_index.append(top_3_df)
    return edge_index


def create_edge_index(edge, state_to_idx):
    indexes = []
    edges = copy.deepcopy(edge)
    for df in edges:
        source = df['origin_state'].map(state_to_idx).values
        target = df['destination_state'].map(state_to_idx).values
        edge_index = torch.tensor([source, target], dtype=torch.long)
        indexes.append(edge_index)
    return indexes



import copy
def edge_features(edge_index, nodes, state_to_idx):
    new_edge_index = []
    new_nodes = []
    for df in edge_index:
        x = copy.deepcopy(df)
        x['destination_index'] = x['destination_state'].map(state_to_idx)
        x['origin_index'] = x['origin_state'].map(state_to_idx)
        new_edge_index.append(x)
    for year in nodes:
        y = copy.deepcopy(nodes[year])
        y['state_index'] = y.index.map(state_to_idx)
        new_nodes.append(y)
    return new_edge_index, new_nodes
    

def merge_edge(edges,nodes):
    processed_dfs = []
    for i, edge_df in enumerate(edges):
        df = edge_df.copy()
        nodes_df = nodes[i]
        nodes_df.set_index('state_index', inplace=True)
        merged_df = pd.merge(
            df, 
            nodes_df, 
            left_on='destination_index', 
            right_index=True, # Merge using the integer index of nodes_df
            how='left'
        )
        rename_mapping = {col: f'{col}_D' for col in nodes_df.columns}
        merged_df.rename(columns=rename_mapping, inplace=True)
        
        merged_df2 = pd.merge(
            merged_df, 
            nodes_df, 
            left_on='origin_index', 
            right_index=True, # Merge using the integer index of nodes_df
            how='left'
        )
        rename_mapping = {col: f'{col}_O' for col in nodes_df.columns}
        merged_df2.rename(columns=rename_mapping, inplace=True)
        processed_dfs.append(merged_df2)
    
    return processed_dfs

def edgefeature_engineering(features):
    final = []
    for df in features:
        #df['income_ratio'] = df['personal_income_D'] / df['personal_income_O']
        #df['gdp_ratio'] = df['gdp_D'] / df['gdp_O']
        #df['median_rent_ratio'] = df['median_rent_D'] / df['median_rent_O']
        #df['median_homevalue_ratio'] = df['median_homevalue_D'] / df['median_homevalue_O']
        #df['unemployment_rate_diff'] = df['unemployment_rate_D'] - df['unemployment_rate_O']
        df['median_income_ratio'] = df['median_household_income_D'] / df['median_household_income_D']
        #df['mean_income_ratio'] = df['mean_household_income_D'] / df['mean_household_income_O']

        #final.append(df[['gdp_ratio','median_rent_ratio','median_homevalue_ratio','unemployment_rate_diff','median_income_ratio','mean_income_ratio']])
        final.append(df[['median_income_ratio']])
    return final
def standardize_edge_features(list_of_tensors,train_years):
    
    train_tensors = [t for i, t in enumerate(list_of_tensors) if (2005 + i) in train_years]
    
    scaler = StandardScaler()
    scaler.fit(torch.cat(train_tensors))
    newtensors = []
    for tensor in list_of_tensors:
        tensor = scaler.transform(tensor)
        tensor = torch.tensor(tensor, dtype=torch.float32)
        newtensors.append(tensor)
    
    return newtensors,scaler
def edge_feature_total(edge_index, nodes, state_to_idx,train_years):
    e,n = edge_features(edge_index, nodes, state_to_idx)
    features = merge_edge(e,n)
    final = edgefeature_engineering(features)
    edge_tensors = [torch.tensor(df.values, dtype=torch.float32) for df in final]
    
    edge_tensors,scaler_edge_feature = standardize_edge_features(edge_tensors, train_years)
    return edge_tensors, scaler_edge_feature