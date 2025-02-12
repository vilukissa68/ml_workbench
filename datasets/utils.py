import requests
import tarfile


def download_file(url, local_filename):
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        with open(local_filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

    return local_filename


def extract_tar_gz(file_path, extract_to="."):
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=extract_to)
    print(f"Extracted {file_path} to {extract_to}")
