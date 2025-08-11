


import pandas as pd
import requests
import time
from functools import reduce
import concurrent.futures
import threading
from queue import Queue

def fetch_acs_data(start_year, end_year, dataset_profiles, max_workers=20):
    api_key = 'a1f6fe95b425fcdea61048e5e94ecb1b9d879a53'
    final_yearly_dataframes = {}
    
    def fetch_single_request(year, profile):
        """Fetch a single profile for a single year"""
        base_url = f"https://api.census.gov/data/{year}/acs/acs1/profile"
        params = {'get': f"group({profile})", 'for': 'state:*', 'key': api_key}
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return year, profile, pd.DataFrame(data[1:], columns=data[0])
        except Exception as e:
            print(f"  > FAILED to fetch {profile} for {year}. Error: {e}")
            return year, profile, None
    
    # Create all year/profile combinations
    tasks = [(year, profile) for year in range(start_year, end_year + 1) 
             if year != 2020
             for profile in dataset_profiles]
    
    # Execute requests in parallel
    results_by_year = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {executor.submit(fetch_single_request, year, profile): (year, profile) 
                         for year, profile in tasks}
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_task):
            year, profile, df = future.result()
            if df is not None:
                if year not in results_by_year:
                    results_by_year[year] = []
                results_by_year[year].append(df)
    
    # Merge DataFrames for each year
    for year, dfs_list in results_by_year.items():
        if len(dfs_list) > 1:
            print(f"  > Merging {len(dfs_list)} profile(s) for {year}...")
            merged_df = reduce(lambda left, right: pd.merge(left, right, on=['NAME', 'state']), dfs_list)
            final_yearly_dataframes[year] = merged_df
        elif len(dfs_list) == 1:
            final_yearly_dataframes[year] = dfs_list[0]
    
    return final_yearly_dataframes


def remove_columns(data_dict):
    """
    Removes columns ending in 'EA' or 'MA' from a dictionary of DataFrames.
    Also removes Puerto Rico row.
    
    Args:
        data_dict (dict): Dictionary with years as keys and DataFrames as values
        
    Returns:
        dict: Dictionary of DataFrames with EA/MA columns removed and Puerto Rico filtered out
    """
    import copy
    
    # Create a deep copy to avoid modifying the original dataframes
    dict_copy = copy.deepcopy(data_dict)
    
    for year, df in dict_copy.items():
        # Remove Puerto Rico row BEFORE setting index - make explicit copy
        df = df[df['NAME'] != 'Puerto Rico'].copy()  # Add .copy() here
        
        # Set NAME as index
        df.set_index('NAME', inplace=True, drop=True)
        df.index.name = 'NAME'  # Ensure index name is set
        
        # Find columns to drop
        columns_to_drop = []
        for col in df.columns:
            if col.endswith('EA') or col.endswith('MA') or col.endswith('PM') or col.endswith('M') or col.endswith('SS'):
                columns_to_drop.append(col)
            # Check if DataFrame is not empty before accessing first row
            if len(df) > 0 and str(df[col].iloc[0]) == '-888888888':  # Convert to string for comparison
                columns_to_drop.append(col)
        
        # Drop the columns
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
        else:
            print(f"Year {year}: No columns ending in EA/MA found")
        
        # Update the dictionary with the cleaned DataFrame
        dict_copy[year] = df

    return dict_copy

def fetch_variable_labels(data_dict):
    """
    Fetches variable labels from Census API and maps column names to human-readable labels
    Uses parallel processing and caching for faster execution.
    
    Args:
        data_dict (dict): Dictionary with years as keys and DataFrames as values
        
    Returns:
        dict: Dictionary with years as keys and DataFrames with labeled columns as values
    """
    import copy
    import concurrent.futures
    from functools import lru_cache
    
    # Create a deep copy to avoid modifying the original dictionary
    mapped_dict = copy.deepcopy(data_dict)
    
    @lru_cache(maxsize=50)
    def fetch_variables_for_year(year):
        """Cached function to fetch variables for a specific year"""
        api_url = f"https://api.census.gov/data/{year}/acs/acs1/profile/variables.json"
        
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            
            # Load the JSON data
            variables_data = response.json()
            variables_dict = variables_data['variables']
            
            # Create a dictionary mapping variable IDs to their labels
            variable_mapping = {key: value['label'] for key, value in variables_dict.items() 
                              if not key.startswith('GEO_ID')}
            
            return variable_mapping
            
        except Exception as e:
            print(f"  Error fetching variables for year {year}: {e}")
            return {}
    
    def process_year(year_df_pair):
        """Process a single year's DataFrame"""
        year, df = year_df_pair
        
        # Get variable mapping for this year (cached)
        variable_mapping = fetch_variables_for_year(year)
        
        if not variable_mapping:
            return year, df  # Return unchanged if API call failed
        
        # Build column renaming dictionary
        column_rename_dict = {}
        
        for col in df.columns:
            if col in variable_mapping:
                column_rename_dict[col] = variable_mapping[col]
        
        # Rename columns
        if column_rename_dict:
            df_copy = df.copy()
            df_copy.rename(columns=column_rename_dict, inplace=True)
            return year, df_copy
        
        return year, df
    
    # Process all years in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        future_to_year = {
            executor.submit(process_year, (year, df)): year 
            for year, df in mapped_dict.items()
        }
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_year):
            year = future_to_year[future]
            try:
                processed_year, processed_df = future.result()
                mapped_dict[processed_year] = processed_df
            except Exception as e:
                print(f"  Error processing year {year}: {e}")
    
    return mapped_dict

def clean_column_prefixes(data_dict):
    """
    Removes 'Percent!!' or 'Estimate!!' prefixes from column names in a dictionary of DataFrames.
    
    Args:
        data_dict (dict): Dictionary with years as keys and DataFrames as values
        
    Returns:
        dict: Dictionary of DataFrames with cleaned column names
    """
    import copy
    
    # Create a deep copy to avoid modifying the original dataframes
    dict_copy = copy.deepcopy(data_dict)
    
    for year, df in dict_copy.items():
        # Create dictionary for column renaming
        column_rename_dict = {}
        
        for col in df.columns:
            new_col = col
            
            # Remove 'Percent!!' prefix and add 'P' at start
            if col.startswith('Number!!'):
                new_col = new_col.replace('Number!!', '', 1)
                column_rename_dict[col] = new_col
            if 'Percent!!' in col:
                new_col =  new_col.replace('Percent!!', '', 1) + "P"
                column_rename_dict[col] = new_col
            if 'Estimate!!' in col:
                new_col = new_col.replace('Estimate!!', '', 1)
                column_rename_dict[col] = new_col
        
        # Apply all column renames at once
        if column_rename_dict:
            df.rename(columns=column_rename_dict, inplace=True)
        else:
            print(f"Year {year}: No columns with 'Percent!!' or 'Estimate!!' prefixes found")

    return dict_copy

def fetch_process_acs(start,end, profiles):
    
    # Remove EA and MA columns
    fetch = fetch_acs_data(start, end, profiles)
    print(f"Finished fetching!")
    processed = remove_columns(fetch)
    print("Finished removing extra columns")
    processed = fetch_variable_labels(processed)
    print("Finished fetching variable labels")
    processed = clean_column_prefixes(processed)
    print("Finished cleaning column prefixes")




    #time.sleep(1) 
    print("\n=== Processing Complete")
    

    return processed




