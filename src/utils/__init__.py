def add_root_to_path():
    import pathlib, sys
    root_folder = pathlib.Path(__file__).parents[2].__str__()
    if root_folder not in sys.path:
        sys.path.append(root_folder)
