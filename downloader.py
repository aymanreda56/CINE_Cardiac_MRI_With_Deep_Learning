from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TaipingQu/CMR-MULTI",
    repo_type="dataset",
    local_dir="CMR-MULTI",
    local_dir_use_symlinks=False
)
