from huggingface_hub import HfApi

username = "UtkarshRai14"
repo_name = "movie_recommender"

files_to_upload = [
    "similarity.pkl.gz",
    "movies.pkl"
]

api = HfApi()

for file in files_to_upload:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,  # Upload with same name in repo
        repo_id=f"{username}/{repo_name}",
        repo_type="model"
    )
    print(f"✅ Uploaded: {file}")

print(f"\n🎉 All files uploaded to: https://huggingface.co/{username}/{repo_name}")
