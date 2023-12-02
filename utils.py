import subprocess
from importlib.metadata import distribution, PackageNotFoundError
import zipfile
from pathlib import Path

from IPython.display import clear_output

import random
import numpy as np
import torch

def pip_install(package: str):
    try:
        # Проверка, установлен ли пакет
        distribution(package)
        print(f"Пакет '{package}' уже установлен.")
    except PackageNotFoundError:
        # Установка пакета, если он не найден
        process = subprocess.Popen(["pip", "install", package], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(process.stdout.readline, b''):
            print(line.decode(), end='')
        # clear_output() # Эту строку следует использовать в контексте Jupyter Notebook
        print(f"Пакет '{package}' успешно установлен.")


def fix_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True    
    
    
def unzip_file(zip_path, dest_folder):
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    
    print(f"Архив '{zip_path}' распакован в '{dest_folder}'")
    

def unrar_file(rar_path, dest_folder):

    pip_install('rar')
    dest_folder.mkdir(parents=True, exist_ok=True)

    process = subprocess.Popen(['unrar', 'x', rar_path, dest_folder],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               text=True)

    file_count = 0
    try:
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if 'Extracting' in output:
                file_count += 1
                print(f"\rРаспаковано файлов: {file_count}", end="")

    except KeyboardInterrupt:
        # Обработка прерывания пользователем
        print("\nПроцесс прерван пользователем.")
        return

    print(f"\rВсего распаковано файлов: {file_count}")
    

