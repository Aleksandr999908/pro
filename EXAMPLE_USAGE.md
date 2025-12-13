# Примеры использования

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Подготовка данных

Разместите видео в структуре:
```
data/videos/
  ├── 2024-10/
  │   ├── fox/
  │   │   └── video1.mp4
  │   └── wolf/
  │       └── video2.mp4
```

### 3. Извлечение кадров

```bash
python scripts/extract_frames.py \
    --video_dir data/videos \
    --output_dir data/frames \
    --fps 1.0
```

### 4. Обучение модели

```bash
python scripts/train.py \
    --config models/config.yaml \
    --data_dir data/roi \
    --output_dir models/checkpoints
```

### 5. Экспорт модели в ONNX

```bash
python scripts/export_onnx.py \
    --model models/checkpoints/best_model.pth \
    --config models/config.yaml \
    --output models/fgc_day.onnx
```

### 6. Запуск API сервиса

```bash
python service/app.py
```

API будет доступен на http://localhost:8000

### 7. Тестирование API

```bash
# Через curl
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_encoded_image>",
    "domain": "day"
  }'

# Через Python
import requests
import base64
from PIL import Image

image = Image.open("test_image.jpg")
buffer = io.BytesIO()
image.save(buffer, format='JPEG')
img_base64 = base64.b64encode(buffer.getvalue()).decode()

response = requests.post(
    "http://localhost:8000/classify",
    json={"image": img_base64, "domain": "day"}
)
print(response.json())
```

### 8. Интеграция с EcoAssist

```bash
python integrations/ecoassist_postprocess.py \
    --input_json megadetector_result.json \
    --output_json refined_result.json \
    --image frame.jpg \
    --api_url http://localhost:8000 \
    --domain day
```

## Использование Docker

### Запуск через Docker Compose

```bash
docker-compose up -d
```

### Запуск через Docker

```bash
# Сборка образа
docker build -t fox-wolf-classifier -f service/Dockerfile .

# Запуск контейнера
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    fox-wolf-classifier
```

## Примеры API запросов

### POST /classify

```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAA...",
  "domain": "day",
  "metadata": {
    "month": "2024-10",
    "location": "camera_1"
  }
}
```

### POST /classify/file

```bash
curl -X POST http://localhost:8000/classify/file \
  -F "file=@test_image.jpg" \
  -F "domain=day"
```

## Временная агрегация

```bash
python scripts/temporal_aggregation.py \
    --input classifications.json \
    --output aggregated.json \
    --method majority_vote \
    --window_size 5
```

## Интеграция в Python код

```python
from integrations.ecoassist_postprocess import FoxWolfRefiner

refiner = FoxWolfRefiner(api_url="http://localhost:8000")

# Классификация одного изображения
from PIL import Image
image = Image.open("frame.jpg")
result = refiner.classify(image, domain="day")
print(result)

# Обработка результата MegaDetector
import json
with open("megadetector_result.json") as f:
    md_result = json.load(f)

refined = refiner.process_megadetector_result(
    md_result,
    image_path="frame.jpg",
    domain="day"
)

print(json.dumps(refined, indent=2))
```
