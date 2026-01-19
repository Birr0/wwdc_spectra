from datasets import load_dataset
ds = load_dataset(
    "parquet",
    data_files=f"/data/dtce-schmidt/phys2526/sdss_II/spender_I_flow/embeddings_flowmatch/6997867_0/test/*.parquet",
    split="train"
)

print(ds)
print(ds["orig"][0])
print(ds["cond"][1])
print(ds["uncond"][3])
print(ds["id"][10])