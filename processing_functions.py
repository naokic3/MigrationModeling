import copy 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def processing(bea, acs,vars):
    """
    Complete processing pipeline.
    
    Parameters:
    bea (dict): BEA data dictionary
    acs (dict): ACS data dictionary  
    feature_list (list): List of feature names to extract
    
    Returns:
    dict: Processed and normalized data ready for modeling
    """
    bea = copy.deepcopy(bea)
    acs = copy.deepcopy(acs)
    data = combine_acs_bea(acs, bea)

    data = choose_target_features(data,vars)
    data = convert(data)
    check_consistent_columns(data)

    return data


def combine_acs_bea(acs, bea):
    """
    Combine ACS and BEA dictionaries of dataframes.
    
    Parameters:
    acs (dict): Dictionary with years as keys and DataFrames as values
    bea (dict): Dictionary with years as keys and DataFrames as values
    
    Returns:
    dict: Combined dictionary with merged DataFrames for each year
    """
    import pandas as pd
    
    combined = {}
    
    # Get all unique years from both dictionaries
    all_years = set(acs.keys()) | set(bea.keys())
    
    for year in all_years:

        if year in acs and year in bea:
            # FIXED: Both datasets have 'NAME' as index - merge on index
            combined[year] = pd.merge(acs[year], bea[year], 
                                    left_index=True, right_index=True,
                                    how='outer', suffixes=('_acs', '_bea'))
        elif year in acs:
            # Only ACS has data for this year
            combined[year] = acs[year].copy()
        elif year in bea:
            # Only BEA has data for this year
            combined[year] = bea[year].copy()
        combined[year] = combined[year][~combined[year].index.str.contains('District of Columbia', case=False, na=False)]
    return combined

def remove_error_totals(data_dict):
    """
    Process columns to label them with P (percent) or E (estimate) suffixes.
    Removes error columns and cleans up column names.
    
    Parameters:
    data_dict (dict): Dictionary with years as keys and DataFrames as values
    
    Returns:
    dict: Single dictionary with cleaned columns labeled with P or E suffixes
    """
    processed_dict = copy.deepcopy(data_dict)

    for year in processed_dict.keys():
        df = processed_dict[year]
        columns_to_drop = []
        columns_to_rename = {}
        
        for col in df.columns:
            # Drop error columns
            if col.endswith('Error'):
                columns_to_drop.append(col)
            
            # Process percent columns
            elif col.endswith('Percent') or '!!Percent' in col or col.startswith('Percent Estimate!!'):
                # Clean the column name and add P suffix
                clean_name = col.replace('!!Percent', '').replace('Percent', '')
                new_name = f"{clean_name}P"
                columns_to_rename[col] = new_name
            
            # Process estimate columns  
            elif col.endswith('Estimate') or '!!Estimate' in col or col.startswith('Estimate!!'):
                # Clean the column name and add E suffix
                clean_name = col.replace('!!Estimate', '').replace('Estimate', '')
                new_name = f"{clean_name}E"
                columns_to_rename[col] = new_name
        
        # Apply changes
        df.drop(columns=columns_to_drop, inplace=True)
        df.rename(columns=columns_to_rename, inplace=True)
        
        processed_dict[year] = df

    return processed_dict

def choose_target_features(census, column_mapping):
    """
    Extract and rename target features from census data dictionary
    
    Parameters:
    census (dict): Dictionary with years as keys and DataFrames as values
    column_mapping (dict): Dictionary mapping search terms to new names
                          {'search_term': 'new_feature_name'}
    
    Returns:
    dict: Dictionary with years as keys and DataFrames with selected and renamed features
    """
    census_target = {}
    
    # Initialize empty DataFrames for each year with same index as original
    for year in census.keys():
        census_target[year] = pd.DataFrame(index=census[year].index)
    
    for search_term, new_feature_name in column_mapping.items():

        years_found = []
        
        # Search for matching column in EACH year individually
        for year in census.keys():
            df = census[year]
            
            # Search for columns that contain the search term (case insensitive)
            matching_cols = [col for col in df.columns 
                           if search_term.lower() in col.lower()]
            
            if matching_cols:
                # Use the first match for this year
                col_to_use = matching_cols[0]
                census_target[year][new_feature_name] = df[col_to_use]
                years_found.append(year)
    census_target = convert(census_target)
    return census_target

def rename_columns_in_dict(data_dict, column_mapping):
    """
    Helper function to rename columns in a dictionary of DataFrames.
    
    Parameters:
    data_dict (dict): Dictionary with years as keys and DataFrames as values
    column_mapping (dict): Dictionary mapping old column names to new column names
    
    Returns:
    dict: Dictionary with DataFrames containing renamed columns
    """
    import copy
    
    # Create a deep copy to avoid modifying the original dataframes
    renamed_dict = copy.deepcopy(data_dict)
    
    for year, df in renamed_dict.items():
        # Find which columns from the mapping exist in this DataFrame
        existing_mappings = {}
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                existing_mappings[old_name] = new_name
            else:
                # Try case-insensitive matching
                matching_cols = [col for col in df.columns if old_name.lower() in col.lower()]
                if matching_cols:
                    # Use the first match
                    existing_mappings[matching_cols[0]] = new_name
                    print(f"Year {year}: Matched '{old_name}' to column '{matching_cols[0]}'")
        
        # Rename the columns
        if existing_mappings:
            df.rename(columns=existing_mappings, inplace=True)
            print(f"Year {year}: Renamed {len(existing_mappings)} columns")
    
    return renamed_dict

def convert(census_target):
    """
    Convert object columns to float in a dictionary.
    
    Parameters:
    census_target (dict): Dictionary with years as keys and DataFrames as values
    
    Returns:
    dict: Dictionary with converted DataFrames (all numeric columns as float64)
    """
    converted_dict = {}
    
    for year, df in census_target.items():
        df_copy = df.copy()  # Make a copy to avoid modifying original
        
        # Convert all columns to float
        for col in df_copy.columns:
            # Handle common string issues before conversion
            df_copy[col] = pd.to_numeric(
                df_copy[col].astype(str).str.replace(',', '').replace('-', '0'), 
                errors='coerce'
            ).astype('float64')  # Explicitly convert to float64
        
        converted_dict[year] = df_copy
    
    return converted_dict

def normalize(data_dict, method='z-score'):
    """
    Normalize data in dictionary of DataFrames.
    
    Parameters:
    data_dict (dict): Dictionary with years as keys and DataFrames as values
    method (str): 'z-score' or 'min-max'
    
    Returns:
    dict: Dictionary with normalized DataFrames
    """
    normalized_dict = {}
    
    for year, df in data_dict.items():
        if method == 'z-score':
            scaler = StandardScaler()
        elif method == 'min-max':
            scaler = MinMaxScaler()
        else:
            raise ValueError("method must be 'z-score' or 'min-max'")
        
        # Fit and transform the data
        scaled_data = scaler.fit_transform(df)
        
        # Create new DataFrame with scaled data
        normalized_dict[year] = pd.DataFrame(
            scaled_data, 
            columns=df.columns, 
            index=df.index
        )
    
    return normalized_dict

def dict_to_hdf5(yearly_dataframes, output_filename='Model_Data.h5'):
    """
    Save dictionary of DataFrames to HDF5 format.
    
    Args:
        yearly_dataframes (dict): Dictionary where keys are years and values are DataFrames
        output_filename (str): Name of the output HDF5 file
    
    Returns:
        str: Path to the saved file
    """
    import os
    
    if not yearly_dataframes:
        raise ValueError("Input dictionary is empty")
    
    # Add .h5 extension if not present
    if not output_filename.endswith('.h5'):
        output_filename = f"{output_filename}.h5"
    
    # Save to HDF5
    with pd.HDFStore(output_filename, mode='w') as store:
        for year, df in yearly_dataframes.items():
            # Use year as the key in HDF5 store
            store[f'year_{year}'] = df
            
        # Store metadata
        metadata = pd.DataFrame({
            'years': list(yearly_dataframes.keys()),
            'num_rows': [len(df) for df in yearly_dataframes.values()],
            'num_cols': [len(df.columns) for df in yearly_dataframes.values()]
        })
        store['metadata'] = metadata
    
    # Calculate file size
    file_size_mb = os.path.getsize(output_filename) / 1024 / 1024
    total_rows = sum(len(df) for df in yearly_dataframes.values())
    
    print(f"Saved {len(yearly_dataframes)} years of data to {output_filename}")
    print(f"Total rows: {total_rows:,}")
    print(f"File size: {file_size_mb:.2f} MB")
    
    return output_filename

def load_hdf5(filename='Model_Data.h5'):
    """
    Simple function to load HDF5 file back to dictionary of DataFrames.
    
    Args:
        filename (str): Name of the HDF5 file
    
    Returns:
        dict: Dictionary with years as keys and DataFrames as values
    """
    data_dict = {}
    
    with pd.HDFStore(filename, mode='r') as store:
        for key in store.keys():
            if key.startswith('/year_'):
                year = int(key.replace('/year_', ''))
                data_dict[year] = store[key]
    
    return data_dict

def check_consistent_columns(data_dict):
    """
    Check if all DataFrames in a dictionary have the same columns.
    Prints "Done" if consistent, otherwise shows differences.
    
    Parameters:
    data_dict (dict): Dictionary with years as keys and DataFrames as values
    
    Returns:
    bool: True if consistent, False otherwise
    """
    if not data_dict:
        print("Done")
        return True
    
    # Get all years and their column sets
    years = sorted(data_dict.keys())
    column_sets = {year: set(data_dict[year].columns) for year in years}
    
    # Find common columns (intersection of all)
    common_columns = set.intersection(*column_sets.values())
    
    # Find all unique columns (union of all)
    all_columns = set.union(*column_sets.values())
    
    # Check if all DataFrames have the same columns
    is_consistent = len(common_columns) == len(all_columns)
    
    if is_consistent:
        print("Complete. No errors found.")
        return True
    
    # Show differences
    print("INCONSISTENT COLUMNS FOUND:")
    
    # Show column counts for each year
    print("\nColumn counts by year:")
    for year in years:
        print(f"  {year}: {len(column_sets[year])} columns")
    
    # Show which columns are missing from which years
    print(f"\nColumns missing from some years:")
    missing_from_years = {}
    for col in all_columns:
        years_missing = [year for year in years if col not in column_sets[year]]
        if years_missing:
            missing_from_years[col] = years_missing
    
    for col, missing_years in missing_from_years.items():
        print(f"  '{col}' missing from: {missing_years}")
    
    return False

def convert_to_percentages(data_dict, columns_to_convert, base_column='pop_25+', years=None):

    import copy
    
    # Create a deep copy to avoid modifying original data
    converted_dict = copy.deepcopy(data_dict)
    
    # Determine which years to process
    if years is None:
        years_to_process = converted_dict.keys()
    else:
        # Only process years that exist in both the data and the years parameter
        years_to_process = [year for year in years if year in converted_dict]
    
    for year in years_to_process:
        df = converted_dict[year]
        
        # Check if base column exists
        if base_column not in df.columns:
            print(f"Year {year}: Base column '{base_column}' not found. Skipping year.")
            continue
            
        converted_cols = []
        missing_cols = []
        
        for col in columns_to_convert:
            if col in df.columns:
                # Convert to percentage (multiply by 100) and round to 1 decimal place
                df[col] = ((df[col] / df[base_column]) * 100).round(1)
                converted_cols.append(col)
            else:
                missing_cols.append(col)
        
        # Report results
        if converted_cols:
            print(f"Year {year}: Converted {len(converted_cols)} columns to percentages: {converted_cols}")
        
        if missing_cols:
            print(f"Year {year}: Missing columns: {missing_cols}")
    
    return converted_dict

def load_yearly_excel_files(file_pattern, start_year, end_year, sheet_name=0):
    """
    Load Excel files with year-based naming pattern into a dictionary of DataFrames.
    Automatically tries both .xlsx and .xls extensions.
    
    Args:
        file_pattern (str): Base file pattern (year will be added at the end)
        start_year (int): Starting year
        end_year (int): Ending year (inclusive)
        sheet_name (str/int): Sheet name or index to load (default: 0)
    
    Returns:
        dict: Dictionary with years as keys and DataFrames as values
    """
    import pandas as pd
    import os
    
    yearly_data = {}
    successful_loads = []
    failed_loads = []
    
    for year in range(start_year, end_year + 1):
        if year == 2020:
            continue
        loaded = False
        
        # Check if file pattern already has extension
        if file_pattern.endswith('.xlsx') or file_pattern.endswith('.xls'):
            # Extension already in pattern, just add year before extension
            base_name, ext = os.path.splitext(file_pattern)
            filename = f"{base_name}{year}{ext}"
            filenames_to_try = [filename]
        else:
            # No extension in pattern, try both extensions
            base_filename = f"{file_pattern}{year}"
            extensions = ['.xlsx', '.xls']
            filenames_to_try = [base_filename + ext for ext in extensions]
        
        for filename in filenames_to_try:
            try:
                if os.path.exists(filename):
                    df = pd.read_excel(filename, sheet_name=sheet_name)
                    yearly_data[year] = df
                    successful_loads.append(year)
                    
                    loaded = True
                    break
                    
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
        
        if not loaded:
            failed_loads.append(year)
            print(f"✗ No file found for {year}")
    
    return yearly_data


def remove(data):
    dict = copy.deepcopy(data)
    for year, df in dict.items():
        first = df.columns[0]
        df = df.rename(columns={first: 'NAME'})    
        df.set_index('NAME', inplace=True)
        cols_to_drop = []
        cols_to_rename = {}

        for col in df.columns:
            # Check if column should be dropped
            if 'Table' in str(df.iloc[0][col]):
                cols_to_drop.append(col)
            # Check if column should be renamed
            elif pd.notna(df.iloc[5][col]):
                cols_to_rename[col] = df.iloc[5][col]

        df = df.rename(columns=cols_to_rename)
        df = df.drop(columns=cols_to_drop)


        cols_to_drop = df.columns[1::2]
        df = df.drop(columns=cols_to_drop)
        df = df.dropna(subset=[df.columns[0]])

        
        

        if year >= 2010:
            df = df.drop(df.columns[[0, 1,2, 3]], axis=1)
            
        # Get columns to drop based on first row values
        

        


        df = df[df.index.notna()]
        df = df[~df.index.str.contains('current residence', case=False, na=False)]
        df = df[~df.index.str.contains('District of Columbia', case=False, na=False)]
        df = df[~df.index.str.contains('United States', case=False, na=False)]
        df = df[~df.index.str.contains('Puerto Rico', case=False, na=False)]
        df = df.loc[:, ~df.columns.str.contains('District of Columbia', case=False, na=False)]
        df = df.loc[:, ~df.columns.str.contains('Puerto Rico', case=False, na=False)]
        df = df.loc[:, ~df.columns.str.contains('Island Area', case=False, na=False)]
        df = df.loc[:, ~df.columns.str.contains('Foreign', case=False, na=False)]
        for col in df.columns:
            if df[col].dtype == 'object':
                # Handle common string issues before conversion
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '').replace('-', '0'), errors='coerce')
        df = df.fillna(0).infer_objects(copy=False)
        # Remove footnotes and everything after
        for i, row in enumerate(df.index):
            if 'Footnotes:' in str(row):
                df = df.iloc[:i]
                break
        dict[year] = df
    return dict 

def migration_processing(file_pattern, start_year, end_year):
    # Load yearly data
    yearly_data = load_yearly_excel_files(file_pattern, start_year, end_year)

    # process
    yearly_data = remove(yearly_data)
    yearly_data = set_diagonal_to_zero(yearly_data)
    yearly_data = convert_dict_to_float64(yearly_data)

    # Further processing can be added here

    return yearly_data

def merge_migration(data,migration):
    #merges two dictionaries of dataframes
    dict = {}
    for year in data.keys():
            dict[year] = pd.concat([data[year], migration[year]], axis=1)
    return dict

def remove_footnotes(data_dict):
    """
    Remove all rows from 'Footnotes:' onwards by checking index names
    
    Parameters:
    data_dict: Dictionary with year keys and DataFrame values
    
    Returns:
    Dictionary with cleaned DataFrames
    """
    cleaned_dict = {}
    
    for year, df in data_dict.items():
        df_copy = df.copy()
        footnotes_idx = None
        
        # Look for 'Footnotes:' in the index names
        for idx in df_copy.index:
            if 'Footnotes:' in str(idx):
                footnotes_idx = idx
                break
        
        # If footnotes row found, keep only rows before it
        if footnotes_idx is not None:
            footnotes_position = df_copy.index.get_loc(footnotes_idx)
            df_copy = df_copy.iloc[:footnotes_position]
            print(f"Year {year}: Removed footnotes and {len(df.index) - len(df_copy.index)} total rows")
        else:
            print(f"Year {year}: No 'Footnotes:' found in index")
        
        cleaned_dict[year] = df_copy
    
    return cleaned_dict

def convert_dict_to_float64(data_dict):
    """
    Convert all columns in all DataFrames within a dictionary to float64.
    Uses vectorized operations for better performance.
    
    Parameters:
    data_dict (dict): Dictionary with DataFrames as values
    
    Returns:
    dict: Dictionary with all DataFrame columns converted to float64
    """
    converted_dict = {}
    
    for key, df in data_dict.items():
        # Convert entire DataFrame to float64 at once
        df_converted = df.apply(pd.to_numeric, errors='coerce').astype('float64')
        converted_dict[key] = df_converted
    
    return converted_dict

def normalize_migration_by_population(migration_dict, population_dict):
    """
    Normalize migration data by dividing each column by corresponding population values.
    
    Parameters:
    migration_dict (dict): Dictionary with years as keys and migration DataFrames as values
    population_dict (dict): Dictionary with years as keys and population Series as values
    
    Returns:
    dict: Dictionary with normalized migration DataFrames
    """
    normalized_migration = {}
    
    for year in migration_dict.keys():
        if year in population_dict:
            df = migration_dict[year].copy()
            pop_series = population_dict[year]
            
            # Normalize each column by dividing by corresponding population value
            for i, col in enumerate(df.columns):
                if i < len(pop_series):
                    df[col] = df[col] / pop_series.iloc[i]
            
            normalized_migration[year] = df
        else:
            # If no population data for this year, keep original
            normalized_migration[year] = migration_dict[year].copy()
    
    return normalized_migration

def set_diagonal_to_zero(data_dict):
    """
    Set diagonal values (i,i) to 0 for all DataFrames in dictionary.
    
    Parameters:
    data_dict (dict): Dictionary with DataFrames as values
    
    Returns:
    dict: Dictionary with modified DataFrames
    """
    result_dict = {}
    
    for key, df in data_dict.items():
        df_copy = df.copy()
        
        # Get minimum of rows and columns to avoid index errors
        min_dim = min(len(df_copy), len(df_copy.columns))
        
        # Set diagonal to 0
        for i in range(min_dim):
            df_copy.iloc[i, i] = 0
            
        result_dict[key] = df_copy
    
    return result_dict