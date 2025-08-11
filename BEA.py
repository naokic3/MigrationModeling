import requests
import pandas as pd
def fetch_BEA(TableName, Geo, LineCode, Start, End):
    # Your 36-character API key
    api_key = "84E5674B-6898-4E3D-93C0-D392713DD6B8"
    start_year = Start
    end_year = End

    year_list = [year for year in range(start_year, end_year + 1) if year != 2020]
    year_range_str = ",".join(map(str, year_list))
    
    # Base URL for the BEA API
    base_url = "https://apps.bea.gov/api/data"

    # Parameters for the state-level API request
    params = {
        'UserID': api_key,
        'Method': 'GetData',
        'DatasetName': 'Regional',
        'TableName': TableName,
        'GeoFips': 'STATE',
        'LineCode' : LineCode,  # Use 'ALL' to get all industries
        'Year': year_range_str,
        'ResultFormat': 'JSON'
    }

    try:
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Get the data in JSON format
        data = response.json()

        # Extract the actual data portion
        if 'BEAAPI' in data and 'Results' in data['BEAAPI']:
            if 'Data' in data['BEAAPI']['Results']:
                data_values = data['BEAAPI']['Results']['Data']
                
                # Convert to a pandas DataFrame
                df = pd.DataFrame(data_values)
                print(f"Successfully retrieved {len(df)} records")
                print(df.head())
                return df
            else:
                print("No data found in API response")
                return pd.DataFrame()
        else:
            # Check for API errors
            if 'BEAAPI' in data and 'Error' in data['BEAAPI']:
                print("Error in API Response:", data['BEAAPI']['Error'])
            else:
                print("Unexpected API response structure")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        print(f"HTTP Error: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error processing response: {e}")
        return pd.DataFrame()

# Usage

def create_bea_database(raw_df, column_name):
    # Make a copy to avoid changing the original DataFrame
    df = raw_df.copy()

    # --- 1. Data Cleaning ---
    # Convert DataValue to a number, removing any commas
    df['DataValue'] = pd.to_numeric(df['DataValue'].astype(str).str.replace(',', ''), errors='coerce')
    
    # List of non-state regions to filter out
    non_states = ['United States', 'New England', 'Mideast', 'Great Lakes', 
                  'Plains', 'Southeast', 'Southwest', 'Rocky Mountain', 'Far West',
                  'Puerto Rico']  # Add Puerto Rico here
    df = df[~df['GeoName'].isin(non_states)]
    
    # Drop rows where DataValue could not be converted
    df.dropna(subset=['DataValue'], inplace=True)

    # --- 2. Create single column DataFrames for each year ---
    yearly_data = {}
    for year, data_for_year in df.groupby('TimePeriod'):
        # Create DataFrame with states as index and single named column
        year_df = data_for_year.set_index('GeoName')[['DataValue']].copy()
        year_df.columns = [column_name]
        year_df.index.name = 'NAME'  # Changed from 'GeoName' to 'NAME'
        yearly_data[int(year)] = year_df
        
    return yearly_data

def fetch_process_bea(TableName, Geo, LineCode, Start, End, column_name):
    """
    Fetches BEA data and processes it into a dictionary of DataFrames.

    Args:
        TableName (str): The name of the BEA table to fetch.
        Geo (str): The geographical level (e.g., 'STATE').
        LineCode (int): The line code for the data.
        Start (int): The start year for the data.
        End (int): The end year for the data.
        column_name (str): The name to give to the data column in the output DataFrames.

    Returns:
        dict: A dictionary of DataFrames indexed by year.
    """
    raw_df = fetch_BEA(TableName, Geo, LineCode, Start, End)
    return create_bea_database(raw_df, column_name)

def combine_bea_dataframes(dataframes_by_year):
    """
    Combine multiple BEA databases (arrays of year-organized dataframes) into a single database.
    
    Parameters:
    dataframes_by_year (list): List of dictionaries where each dictionary has years as keys 
                              and DataFrames as values (output from create_bea_database).
    
    Returns:
    dict: A dictionary where keys are years (int) and values are combined DataFrames 
          with all columns from the input databases.
    """
    if not dataframes_by_year:
        return {}
    
    # Get all unique years across all databases
    all_years = set()
    for db in dataframes_by_year:
        all_years.update(db.keys())
    
    combined_data = {}
    
    for year in sorted(all_years):
        year_dfs = []
        
        # Collect all DataFrames for this year from all databases
        for db in dataframes_by_year:
            if year in db:
                year_dfs.append(db[year])
        
        if year_dfs:
            # Combine all DataFrames for this year using outer join on index (state names)
            combined_df = year_dfs[0]
            for df in year_dfs[1:]:
                combined_df = pd.merge(combined_df, df, left_index=True, right_index=True, how='outer')
            
            combined_data[year] = combined_df
    
    return combined_data