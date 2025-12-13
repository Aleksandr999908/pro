# Документация по ROI и Keypoints

## Обзор

Добавлена поддержка работы с ROI (Region of Interest) масками/координатами и keypoints (ключевыми точками) для видео и изображений.

## Структура данных

### ROI (Region of Interest)

ROI могут быть сохранены в двух форматах:

#### 1. Маски изображений (PNG)
```
data/roi/{video_id}/{frame_idx}.png
```
- Чёрно-белая маска (0 = фон, 255 = ROI)
- Пример: `data/roi/video_001/000123.png`

#### 2. JSON с полигонами
```
data/roi/{video_id}/{frame_idx}.json
```
Формат JSON:
```json
{
  "polygons": [
    [[x1, y1], [x2, y2], [x3, y3], ...],
    [[x1, y1], [x2, y2], ...]
  ],
  "frame": 123
}
```

### Keypoints

Keypoints сохраняются в формате JSON:
```
data/keypoints/{image_id}.json
```

#### Формат для одного объекта:
```json
{
  "image_id": "video_001/000123.jpg",
  "width": 1280,
  "height": 720,
  "model": "fgc_day.onnx",
  "keypoints": [
    { "name": "nose", "x": 512.3, "y": 340.8, "score": 0.94 },
    { "name": "left_ear", "x": 480.1, "y": 310.7, "score": 0.89 }
  ],
  "timestamp": "2025-10-21T12:34:56Z"
}
```

#### Формат для многообъектной сцены:
```json
{
  "image_id": "video_001/000123.jpg",
  "width": 1280,
  "height": 720,
  "model": "fgc_day.onnx",
  "keypoints": [],
  "objects": [
    {
      "id": 1,
      "class": "fox",
      "bbox": [100, 100, 200, 250],
      "keypoints": [
        { "name": "nose", "x": 180.0, "y": 200.0, "score": 0.92 }
      ]
    }
  ],
  "timestamp": "2025-10-21T12:34:56Z"
}
```

## Использование

### ROI утилиты

#### Сохранение ROI-маски:
```python
from scripts.roi_utils import save_roi_mask
import cv2
import numpy as np

# Создаём маску
mask = np.zeros((720, 1280), dtype=np.uint8)
cv2.rectangle(mask, (100, 100), (500, 400), 255, -1)

# Сохраняем
mask_path = save_roi_mask(
    mask=mask,
    video_id="video_001",
    frame_idx=123,
    output_dir="data/roi"
)
```

#### Сохранение ROI-координат (JSON):
```python
from scripts.roi_utils import save_roi_polygons, bbox_to_polygon

# Конвертируем bounding box в полигон
bbox = [100, 100, 400, 300]  # [x, y, w, h]
polygon = bbox_to_polygon(bbox)

# Сохраняем
json_path = save_roi_polygons(
    polygons=[polygon],
    video_id="video_001",
    frame_idx=123,
    output_dir="data/roi"
)
```

#### Извлечение ROI из MegaDetector:
```python
from scripts.roi_utils import extract_roi_from_megadetector
from pathlib import Path

# Извлекаем и сохраняем в формате JSON
json_path = extract_roi_from_megadetector(
    frame_path=Path("data/frames/video_001/frame_000123.jpg"),
    megadetector_json=Path("data/megadetector_results.json"),
    video_id="video_001",
    frame_idx=123,
    format="json"  # или "mask"
)
```

### Keypoints утилиты

#### Сохранение keypoints для одного объекта:
```python
from scripts.keypoints_utils import save_keypoints, create_keypoint

keypoints = [
    create_keypoint("nose", 512.3, 340.8, 0.94),
    create_keypoint("left_ear", 480.1, 310.7, 0.89)
]

json_path = save_keypoints(
    image_id="video_001/000123.jpg",
    keypoints=keypoints,
    width=1280,
    height=720,
    model="fgc_day.onnx"
)
```

#### Сохранение keypoints для многообъектной сцены:
```python
from scripts.keypoints_utils import (
    save_keypoints_multi_object,
    create_object,
    create_keypoint
)

objects = [
    create_object(
        obj_id=1,
        obj_class="fox",
        bbox=[100, 100, 200, 250],
        keypoints=[
            create_keypoint("nose", 180.0, 200.0, 0.92)
        ]
    )
]

json_path = save_keypoints_multi_object(
    image_id="video_001/000123.jpg",
    objects=objects,
    width=1280,
    height=720,
    model="fgc_day.onnx"
)
```

## Модели

### Поддержка моделей для разных доменов

Система поддерживает отдельные модели для дневных и ночных/ИК-сцен:

- **Дневная модель**: `models/fgc_day.onnx`
- **Ночная модель**: `models/fgc_night.onnx`

Модели настраиваются в конфигурационных файлах:
- `models/config.yaml`
- `models/config_fast.yaml`

```yaml
deployment:
  models:
    day: "models/fgc_day.onnx"
    night: "models/fgc_night.onnx"
```

### Использование в API

API автоматически выбирает модель в зависимости от параметра `domain`:

```bash
# Использование дневной модели
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "...",
    "domain": "day"
  }'

# Использование ночной модели
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "image": "...",
    "domain": "night"
  }'
```

## Примеры

Полный пример использования всех функций находится в файле:
```
scripts/example_roi_keypoints.py
```

Запуск примера:
```bash
python scripts/example_roi_keypoints.py
```

## Интеграция с существующим кодом

### Обновление extract_frames.py

Скрипт `extract_frames.py` можно расширить для автоматического сохранения ROI:

```python
from scripts.roi_utils import extract_roi_from_megadetector

# В цикле обработки кадров
for frame_path in frame_files:
    video_id = extract_video_id(frame_path)
    frame_idx = extract_frame_idx(frame_path)

    # Извлекаем ROI из MegaDetector
    if megadetector_json.exists():
        extract_roi_from_megadetector(
            frame_path=frame_path,
            megadetector_json=megadetector_json,
            video_id=video_id,
            frame_idx=frame_idx,
            format="json"  # или "mask"
        )
```

## Примечания

1. **Модель fgc_night.onnx**: Файл модели должен быть добавлен вручную в папку `models/`. Система автоматически загрузит её при первом запросе с `domain="night"`.

2. **Структура директорий**: Директории `data/roi/` и `data/keypoints/` создаются автоматически при первом использовании.

3. **Формат keypoints**: Названия keypoints могут быть любыми (например, "nose", "left_ear", "tail_base"), но рекомендуется использовать стандартизированные названия для совместимости.

4. **Производительность**: Модели загружаются лениво (при первом использовании), что позволяет экономить память, если используется только один домен.


