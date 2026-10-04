from huggingface_hub import snapshot_download

print("Downloading dataset...")

path = snapshot_download(
    repo_id="siddharthmb/article-bias-prediction-random-splits",
    repo_type="dataset",
    local_dir="./dataset"
)

print("Download complete!")
print("Files are in:", path)