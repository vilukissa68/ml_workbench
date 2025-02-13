import tarfile


def extract_tar(file_path, extract_to="."):
    with tarfile.open(file_path, "r") as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted {file_path} to {extract_to}")
