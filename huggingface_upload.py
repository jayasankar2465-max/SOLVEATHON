from huggingface_hub import HfApi

api = HfApi()

# Create repo first
api.create_repo(
    repo_id="Jay2465/distilbert-sentiment",
    repo_type="model",
    exist_ok=True  # avoids error if already created
)

# Then upload
api.upload_folder(
    folder_path="distilbert-sentiment",
    repo_id="Jay2465/distilbert-sentiment",
    repo_type="model"
)

api.create_repo(
    repo_id="Jay2465/t5-absa",
    repo_type="model",
    exist_ok=True  # avoids error if already created
)

api.upload_folder(
    folder_path="t5-absa",
    repo_id="Jay2465/t5-absa",
    repo_type="model"
)