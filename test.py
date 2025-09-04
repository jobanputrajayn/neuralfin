import os
import tarfile
from sentence_transformers import SentenceTransformer

# Define the model name and the directory to save it
model_name = 'all-mpnet-base-v2'
local_model_path = f'./{model_name}'
output_archive_name = f'{model_name}.tar.gz'

print(f"Downloading model: {model_name}...")
# Download the model to the specified local path
# This will create a directory named 'all-MiniLM-L6-v2' containing model files
model = SentenceTransformer(model_name)
model.save(local_model_path)
print(f"Model downloaded to: {local_model_path}")

print(f"Creating .tar.gz archive: {output_archive_name}...")
# Create a .tar.gz archive of the downloaded model directory
with tarfile.open(output_archive_name, "w:gz", compresslevel=9) as tar:
    tar.add(local_model_path, arcname=os.path.basename(local_model_path))
print(f"Archive created at: {output_archive_name}")