from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="pretrained_model.keras",
    path_in_repo="pretrained_model.keras",
    repo_id="Raghava-Ram/brain-tumor-efficientnet",
)