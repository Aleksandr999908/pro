import shutil
from pathlib import Path
import random

# Исходная директория
src_dir = Path("./data/frames")
dest_dir = Path("./data/organized")

# Создаем структуру
(dest_dir / "day" / "month_01").mkdir(parents=True, exist_ok=True)
(dest_dir / "night" / "month_01").mkdir(parents=True, exist_ok=True)

# Копируем папки с номерами в новую структуру
for folder_num in range(1, 23):
    src_folder = src_dir / str(folder_num)
    if src_folder.exists():
        # Для примера: половину в day, половину в night
        if folder_num <= 11:
            dest_path = dest_dir / "day" / "month_01" / f"video_{folder_num}"
        else:
            dest_path = dest_dir / "night" / "month_01" / f"video_{folder_num}"

        # Копируем
        if dest_path.exists():
            shutil.rmtree(dest_path)
        shutil.copytree(src_folder, dest_path)
        print(f"Скопировано: {src_folder} -> {dest_path}")
