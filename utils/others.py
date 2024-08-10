from zipfile import ZipFile


def unzip(zip_path: str, des_path: str):
    with ZipFile(zip_path, 'r') as z_obj:
        z_obj.extractall(path=des_path)
    return des_path
