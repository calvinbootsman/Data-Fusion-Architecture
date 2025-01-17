import pandas as pd
import glob

# Specify the directory containing your CSV files and the common file pattern
file_pattern = "C:/Users/Rianne/Downloads/DFA_Assignment3_2024/DFA_Assignment3_2024/*"

# Name of the file to ignore
file_to_ignore = "C:/Users/Rianne/Downloads/DFA_Assignment3_2024/DFA_Assignment3_2024/tf_data.csv"

# Get a list of all matching CSV files
csv_files = glob.glob(file_pattern)

# Initialize a list to store DataFrames
dataframes = []

# Loop through the files and load them
for file in csv_files:
    if file.endswith(file_to_ignore):
        continue  # Skip the file to ignore
    df = pd.read_csv(file)
    # Append the DataFrame, excluding the first two columns if not already in the list
    if len(dataframes) == 0:
        # Include all columns for the first file
        dataframes.append(df)
    else:
        # Include only the unique columns for subsequent files
        dataframes.append(df.iloc[:, 2:])

# Concatenate all DataFrames along columns (axis=1)
combined_df = pd.concat(dataframes, axis=1)

# Remove duplicate columns (keeping the first occurrence)
combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('combined_output.csv', index=False)

print("Combined file saved as 'combined_output.csv'.")
