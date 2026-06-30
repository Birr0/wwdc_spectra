import os 
import glob


from datasets import load_dataset,  Dataset, DatasetDict
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from astroML.datasets import fetch_sdss_specgals
from natsort import natsorted

load_dotenv()
HF_USERNAME = os.getenv("HF_USERNAME")
DATA_ROOT = os.getenv("DATA_ROOT")

# Downloads and caches a robust subset of the MPA-JHU catalog
data = fetch_sdss_specgals()

print(f"Loaded {data.shape[0]} galaxies.")
print(data['lgm_tot_p50'][:5])  # Median total stellar mass


fp = f"{DATA_ROOT}/sdss_II/spender_I_flow_v2/embeddings/7655991_0"
train_files = natsorted(glob.glob(f"{fp}/train/*.parquet"))
test_files = natsorted(glob.glob(f"{fp}/test/*.parquet"))
val_files = natsorted(glob.glob(f"{fp}/val/*.parquet"))

# 2. Define the file path mapping using the explicitly sorted lists
data_files = {
    "train": train_files,
    "test": test_files,
    "val": val_files
}

ds = load_dataset("parquet", data_files=data_files)

df_sdss = pd.DataFrame(data)
df_sdss['merge_id'] = df_sdss['specObjID'].astype('int64')

# Dictionaries to hold your final data for easy access
merged_dfs = {}
aligned_arrays = {}

# 2. Loop through your three Hugging Face splits
splits = ['train', 'test', 'val']

for split in splits:
    print(f"--- Processing '{split}' split ---")
    
    # Convert the current Hugging Face split to a Pandas DataFrame
    df_spender = ds[split].to_pandas()
    
    # Clean the Hugging Face IDs (strip the 'b', quotes, and spaces)
    df_spender['merge_id'] = df_spender['id'].astype(str).str.extract(r'(\d+)')[0].astype('int64')
    
    # Perform the master merge
    matched_df = pd.merge(df_spender, df_sdss, on='merge_id', how='inner')
    
    # Clean up by dropping the temporary merge column
    matched_df = matched_df.drop(columns=['merge_id'])
    
    # Store the full pandas dataframe in our dictionary
    merged_dfs[split] = matched_df
    
    print(f"Successfully merged {len(matched_df)} galaxies.")

print("All splits successfully matched and aligned!")

# 1. Convert your dictionary of Pandas DataFrames into a HF DatasetDict
hf_dataset_dict = DatasetDict({
    split: Dataset.from_pandas(df) for split, df in merged_dfs.items()
})

# 2. Push to the Hugging Face Hub
# Replace "your-username/your-dataset-name" with your desired repository name
hf_dataset_dict.push_to_hub(f"{HF_USERNAME}/jhu_mpa_embedding_match")
print("Dataset successfully uploaded to the Hugging Face Hub!")