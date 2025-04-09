import os


def get_dataset_path(name):
    for path in [
        f'/path/to/data/root1',
        f'/path/to/data/root2',
        f'/path/to/data/root3',
    ]:
        if os.path.exists(path):
            return path