import kagglehub
import os

current_dir = os.getcwd()
path = kagglehub.dataset_download("himalrana2610/indian-skincare-and-grooming-dataset", path=current_dir)
print("Downloaded to:", path)