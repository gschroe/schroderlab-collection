import pandas as pd
from typing import Dict
import re


def parse_star_file(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    Parse a STAR file and extract all data blocks as Pandas DataFrames.
    
    STAR files (Self-defining Text Archiving and Retrieval) are commonly used in 
    cryo-EM for storing particle and metadata information. This function handles 
    multiple data blocks, each containing a loop with column headers and data rows.
    
    Parameters
    ----------
    filepath : str
        Path to the STAR file to parse.
    
    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with block names as keys and corresponding DataFrames as values.
        Block names have the 'data_' prefix removed (e.g., 'data_particles' -> 'particles').
    
    Examples
    --------
    >>> data_dict = parse_star_file('particles.star')
    >>> particles_df = data_dict['particles']
    >>> optics_df = data_dict['optics']
    """
    
    data_blocks = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for data block declaration
        if line.startswith('data_'):
            block_name = line.replace('data_', '')
            
            # Initialize block data
            columns = []
            data_rows = []
            
            i += 1
            
            # Look for loop_ declaration
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith('loop_'):
                    i += 1
                    break
                elif line and not line.startswith('#'):
                    # Handle cases without loop (single-row data blocks)
                    pass
                i += 1
            
            # Parse column headers
            while i < len(lines):
                line = lines[i].strip()
                
                if line.startswith('_rln'):
                    # Extract column name (remove the numbering comment)
                    # Format: _rlnColumnName #N
                    col_name = line.split('#')[0].strip()
                    columns.append(col_name)
                    i += 1
                elif line and not line.startswith('#'):
                    # End of headers, start of data
                    break
                else:
                    i += 1
            
            # Parse data rows
            while i < len(lines):
                line = lines[i].strip()
                
                # Stop at empty lines or new data blocks or comments
                if not line or line.startswith('data_') or line.startswith('#'):
                    break
                
                # Split the data row and match with columns
                values = line.split()
                
                if len(values) == len(columns):
                    data_rows.append(values)
                
                i += 1
            
            # Create DataFrame if we have data
            if columns and data_rows:
                df = pd.DataFrame(data_rows, columns=columns)
                
                # Try to convert numeric columns
                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except (ValueError, TypeError):
                        # Keep as string if conversion fails
                        pass
                
                data_blocks[block_name] = df
        else:
            i += 1
    
    return data_blocks


# Example usage
if __name__ == '__main__':
    # Parse the STAR file
    star_data = parse_star_file('particles-head60.star')
    
    # Print summary
    print("Parsed blocks:")
    for block_name, df in star_data.items():
        print(f"\n{block_name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First row:\n{df.iloc[0]}")
