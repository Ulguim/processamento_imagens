import kagglehub

# Download latest version
path = kagglehub.dataset_download("saeedehkamjoo/standard-test-images")

print("Path to dataset files:", path)