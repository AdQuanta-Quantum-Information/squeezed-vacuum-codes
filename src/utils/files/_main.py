import os
from os.path import isfile
from os.path import isdir as isfolder
from os.path import sep as foldersep
from pathlib import Path


# Define a set of common file extensions
COMMON_EXTENSIONS = {'.mp4', '.mp3', '.txt', '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.csv', '.json', '.xml', '.html', '.zip', '.tar', '.gz', '.bz2', '.7z'}

def _parse_path(path:Path|str) -> str:
    if isinstance(path, Path):
        return path.absolute().__str__()
    return path

def force_folder_exists(folderpath:str|Path) -> None:
    folderpath = _parse_path(folderpath)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

def get_all_files_fullpath_in_folder(folder_full_path:str) -> list[str]:
    return [folder_full_path+foldersep+filename for filename in get_all_file_names_in_folder(folder_full_path=folder_full_path)]


def get_all_file_names_in_folder(folder_full_path:str) -> list[str]:
    return [file for file in os.listdir(folder_full_path) if not isfolder(folder_full_path+foldersep+file)]


def get_last_file_in_folder(folder_full_path:str, none_if_empty:bool=True)->str:
    file_names = get_all_file_names_in_folder(folder_full_path=folder_full_path)
    if none_if_empty and len(file_names)==0:
        return None  #type: ignore
    return file_names[-1]

def has_extension(filename: str | Path) -> bool:
    path_type = filename if isinstance(filename, Path) else Path(filename)
    return path_type.suffix in COMMON_EXTENSIONS