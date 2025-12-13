# Быстрый старт - Обработка видео

## Структура данных

Ваши видео находятся в:
```
data/videos/лисы-волки/
  ├── 01.2025/
  │   ├── Волк/
  │   │   └── 1.MP4, 2.MP4...
  │   └── Лисица/
  │       └── 1.MP4, 2.MP4...
  ├── 02.2025/
  │   ├── Волк/
  │   └── Лисица/
  └── ...
```

## Шаг 1: Установка зависимостей

```bash
pip install -r requirements.txt
```

## Шаг 2: Извлечение кадров из видео

### Автоматическая обработка (Windows)

```bash
scripts\process_videos.bat
```

### Автоматическая обработка (Linux/Mac)

```bash
bash scripts/process_videos.sh
```

### Ручной запуск

```bash
python scripts/extract_frames.py \
    --video_dir data/videos \
    --output_dir data/frames \
    --fps 1.0 \
    --roi_output data/roi \
    --preserve_structure \
    --process_images
```

**Параметры:**
- `--video_dir`: Директория с видео (по умолчанию: `data/videos`)
- `--output_dir`: Директория для сохранения кадров (по умолчанию: `data/frames`)
- `--fps`: Частота извлечения кадров (по умолчанию: 1.0 кадр/сек)
- `--roi_output`: Директория для ROI кадров (по умолчанию: `data/roi`)
- `--preserve_structure`: Сохранять структуру папок (месяц/класс)
- `--process_images`: Обрабатывать JPG файлы как готовые кадры

## Результат

После выполнения скрипта будут созданы:

### Структура кадров:
```
data/frames/
  ├── 01.2025/
  │   ├── Волк/
  │   │   ├── 1/
  │   │   │   ├── frame_000000.jpg
  │   │   │   └── frame_000001.jpg
  │   │   └── 2/
  │   └── Лисица/
  │       └── ...
  └── 02.2025/
      └── ...
```

### Структура ROI:
```
data/roi/
  ├── 01.2025/
  │   ├── Волк/
  │   │   └── ...
  │   └── Лисица/
  │       └── ...
  └── ...
```

## Дополнительные возможности

### Обработка только видео (без изображений)

```bash
python scripts/extract_frames.py \
    --video_dir data/videos \
    --output_dir data/frames \
    --fps 1.0 \
    --no-process_images
```

### Разная частота извлечения

```bash
# Извлечение каждый 2-й кадр (0.5 fps)
python scripts/extract_frames.py --fps 0.5

# Извлечение каждый кадр (30 fps)
python scripts/extract_frames.py --fps 30
```

## Следующие шаги

После извлечения кадров:

1. **Обучение модели:**
   ```bash
   python scripts/train.py \
       --config models/config.yaml \
       --data_dir data/roi \
       --output_dir models/checkpoints
   ```

2. **Экспорт в ONNX:**
   ```bash
   python scripts/export_onnx.py \
       --model models/checkpoints/best_model.pth \
       --config models/config.yaml \
       --output models/fgc_day.onnx
   ```

3. **Запуск API:**
   ```bash
   python service/app.py
   ```

## Поддержка форматов

Скрипт поддерживает:
- Видео: `.mp4`, `.avi`, `.mov`, `.mkv` (регистронезависимо)
- Изображения: `.jpg`, `.jpeg` (регистронезависимо)

## Примечания

- JPG файлы будут скопированы напрямую в структуру кадров
- Видео будут обработаны с извлечением кадров с заданной частотой
- Структура папок (месяц/класс) сохраняется для удобства работы с датасетом
- Все кадры сохраняются в формате JPG
