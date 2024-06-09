from pathlib import Path
def find_cifar():
    cwd = Path.cwd()
    while 'mlrf_cookiecutter' in cwd.parts and cwd.parts[-1] != 'mlrf_cookiecutter':
        cwd = cwd.parent
        
    target_folder_name = 'cifar-10-batches-py'
    found_folders = [p for p in cwd.rglob(target_folder_name) if p.is_dir()]

    if not found_folders:
        raise FileNotFoundError(f'File \'{target_folder_name}\' not found')
    return found_folders[0]
