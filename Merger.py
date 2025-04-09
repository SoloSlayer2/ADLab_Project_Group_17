import pandas as pd
import glob

# Folder where CSV files are stored
folder_path = r"C:\Data Extraction\Dataset Storehouse\*.csv"  # Adjust this path as needed

# Get a list of all CSV files in the folder
csv_files = glob.glob(folder_path)

# Merge all CSV files
df_list = [pd.read_csv(file, encoding="ISO-8859-1") for file in csv_files]
df_merged = pd.concat(df_list, ignore_index=True)

# Shuffle the dataset
df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the merged and shuffled dataset
output_path = r"C:\Data Extraction\Total_Dataset.csv"
df_merged.to_csv(output_path, index=False, encoding="ISO-8859-1")

print("All CSV files merged and shuffled successfully!")
print(f"Final dataset shape: {df_merged.shape}")
