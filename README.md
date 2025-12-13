# Fox/Wolf Classifier для EcoAssist/AddaxAI

Классификатор "Лиса vs Волк" для уточнения классификации животных на видеокадрах с фотоловушек. Интегрируется в пайплайн EcoAssist/AddaxAI как третья модель после MegaDetector и базовой классификации.

## Описание проекта

Проект реализует легковесный классификатор для различения лис и волков на видеокадрах с фотоловушек (день/ночь, частичное попадание животного в кадр). Модель работает поверх результатов MegaDetector и добавляет уточняющую классификацию в JSON-результат.

### Основные возможности

- ✅ Извлечение кадров из видео (1 fps)
- ✅ Предобработка ROI из MegaDetector с паддингом
- ✅ Сохранение ROI-масок и координат (PNG/JSON)
- ✅ Сохранение keypoints (ключевых точек) для кадров
- ✅ Поддержка многообъектных сцен
- ✅ Легковесная модель (EfficientNet/MobileNet)
- ✅ Поддержка день/ночь доменов (fgc_day.onnx / fgc_night.onnx)
- ✅ Focal Loss и ArcFace Loss для метрического обучения
- ✅ Временная агрегация для видеокадров
- ✅ Экспорт в ONNX для эффективного инференса
- ✅ REST API для интеграции
- ✅ Docker контейнер для развёртывания
- ✅ Интеграция с EcoAssist/AddaxAI

## Структура проекта

```
.
├── data/                    # Данные проекта
│   ├── videos/             # Исходные видео
│   ├── frames/             # Извлечённые кадры
│   ├── roi/                # ROI кадры с паддингом
│   └── keypoints/          # Ключевые точки (опционально)
│
├── models/                  # Модели
│   ├── config.yaml         # Конфигурация модели
│   ├── fgc_day.onnx        # Модель для дневных кадров
│   └── fgc_night.onnx      # Модель для ночных кадров
│
├── service/                 # REST API сервис
│   ├── app.py              # FastAPI приложение
│   ├── Dockerfile          # Docker образ для API
│   └── requirements.txt    # Зависимости API
│
├── integrations/            # Интеграции
│   └── ecoassist_postprocess.py  # Постпроцессор для EcoAssist
│
├── scripts/                 # Утилиты и скрипты
│   ├── extract_frames.py   # Извлечение кадров из видео
│   ├── dataset.py          # Класс датасета
│   ├── train.py            # Обучение модели
│   ├── export_onnx.py      # Экспорт модели в ONNX
│   ├── roi_utils.py        # Утилиты для работы с ROI
│   ├── keypoints_utils.py  # Утилиты для работы с keypoints
│   └── example_roi_keypoints.py  # Примеры использования
│
├── requirements.txt        # Общие зависимости
├── docker-compose.yml      # Docker Compose конфигурация
└── README.md              # Документация

```

## Установка и настройка

### Требования

- Python 3.10+
- CUDA 11.8+ (опционально, для GPU)
- Docker (для контейнеризации)

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Настройка конфигурации

Отредактируйте `models/config.yaml` для настройки параметров обучения и инференса.

## Использование

### 1. Подготовка данных

#### Извлечение кадров из видео

```bash
python scripts/extract_frames.py \
    --video_dir data/videos \
    --output_dir data/frames \
    --fps 1.0 \
    --megadetector_dir data/megadetector_results \
    --roi_output data/roi
```

Скрипт:
- Извлекает кадры с частотой 1 fps
- Применяет результаты MegaDetector для извлечения ROI
- Добавляет паддинг 10-20% вокруг ROI
- Сохраняет кадры в структурированные папки

### 2. Обучение модели

```bash
python scripts/train.py \
    --config models/config.yaml \
    --data_dir data/roi \
    --output_dir models/checkpoints
```

Параметры обучения настраиваются в `models/config.yaml`:
- Архитектура: EfficientNet-V2-S или MobileNetV3-Large
- Функция потерь: Focal Loss, ArcFace или CrossEntropy
- Аугментации: отдельные для день/ночь
- Метрики: F1, Balanced Accuracy, AUROC

### 3. Экспорт модели в ONNX

```bash
python scripts/export_onnx.py \
    --model models/checkpoints/best_model.pth \
    --config models/config.yaml \
    --output models/fgc_day.onnx \
    --input_size 224
```

### 4. Запуск REST API сервиса

#### Локально

```bash
python service/app.py
```

Сервис будет доступен по адресу: `http://localhost:8000`

#### Через Docker

```bash
# Сборка образа
docker build -t fox-wolf-classifier -f service/Dockerfile .

# Запуск контейнера
docker run -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    fox-wolf-classifier
```

#### Через Docker Compose

```bash
docker-compose up -d
```

### 5. Интеграция с EcoAssist

```bash
python integrations/ecoassist_postprocess.py \
    --input_json megadetector_result.json \
    --output_json refined_result.json \
    --image frame.jpg \
    --api_url http://localhost:8000 \
    --domain day
```

Скрипт:
- Читает JSON результат MegaDetector
- Извлекает ROI для каждой детекции животного
- Отправляет ROI в API классификации
- Добавляет поля `refine_species` и `refine_conf` в JSON
- Применяет временную агрегацию для видео

## API документация

### POST /classify

Классифицирует изображение (Base64).

**Запрос:**
```json
{
  "image": "<base64_encoded_image>",
  "domain": "day",  // или "night"
  "metadata": {
    "month": "2025-10",
    "location": "camera_1"
  }
}
```

**Ответ:**
```json
{
  "class": "fox",  // fox, wolf, unknown
  "prob": 0.94,
  "refine_conf": 0.94,
  "probs": {
    "fox": 0.94,
    "wolf": 0.05,
    "unknown": 0.01
  }
}
```

### POST /classify/file

Классифицирует изображение из загруженного файла.

**Запрос:** multipart/form-data с полем `file`

**Ответ:** JSON с результатами классификации

### GET /health

Проверка здоровья сервиса.

## Работа с ROI и Keypoints

### ROI (Region of Interest)

Проект поддерживает сохранение ROI в двух форматах:

1. **Маски изображений (PNG)**: `data/roi/{video_id}/{frame_idx}.png`
2. **JSON с полигонами**: `data/roi/{video_id}/{frame_idx}.json`

Пример использования:
```python
from scripts.roi_utils import save_roi_mask, save_roi_polygons

# Сохранение маски
save_roi_mask(mask, video_id="video_001", frame_idx=123)

# Сохранение координат
polygons = [[[100, 100], [500, 100], [500, 400], [100, 400]]]
save_roi_polygons(polygons, video_id="video_001", frame_idx=123)
```

### Keypoints (Ключевые точки)

Keypoints сохраняются в формате JSON: `data/keypoints/{image_id}.json`

Пример использования:
```python
from scripts.keypoints_utils import save_keypoints, create_keypoint

keypoints = [
    create_keypoint("nose", 512.3, 340.8, 0.94),
    create_keypoint("left_ear", 480.1, 310.7, 0.89)
]

save_keypoints(
    image_id="video_001/000123.jpg",
    keypoints=keypoints,
    width=1280,
    height=720,
    model="fgc_day.onnx"
)
```

Подробная документация: см. `ROI_KEYPOINTS_README.md`

Примеры использования: `scripts/example_roi_keypoints.py`

## Конфигурация

Основные параметры в `models/config.yaml`:

### Модель
- `architecture`: EfficientNet-V2-S или MobileNetV3-Large
- `input_size`: 224 пикселей
- `num_classes`: 3 (fox, wolf, unknown)

### Обучение
- `batch_size`: 32
- `num_epochs`: 50
- `learning_rate`: 0.001
- `loss`: focal, arcface или crossentropy

### Инференс
- `confidence_threshold`: 0.7 (порог уверенности)
- `class_gap_threshold`: 0.2 (минимальный зазор между классами)
- `temporal_window`: 5 (окно для временной агрегации)

### Модели
- `deployment.models.day`: путь к модели для дневных сцен (по умолчанию: `models/fgc_day.onnx`)
- `deployment.models.night`: путь к модели для ночных/ИК-сцен (по умолчанию: `models/fgc_night.onnx`)

## Приёмочные критерии

- ✅ F1 ≥ 0.92 для каждого класса
- ✅ Снижение ручной разметки ≥ 30%
- ✅ Отчёты по день/ночь, сезонности и частичности
- ✅ Latency < 50ms на CPU (на кадр)
- ✅ Throughput > 20 fps на CPU

## Метрики и валидация

Проект включает:
- F1 score по классам
- Balanced Accuracy
- AUROC
- Калибровка вероятностей (Temperature Scaling)
- Конфьюжн матрицы
- PR/ROC кривые

## Лицензии

Проект использует открытые компоненты:
- MegaDetector (MIT)
- PyTorch/Timm (Apache-2.0/BSD-3)
- ONNX Runtime (MIT)
- OpenCV (Apache-2.0)

Все зависимости совместимы с коммерческим использованием.

## Разработка

### Структура кода

- `scripts/dataset.py`: Класс датасета для PyTorch
- `scripts/train.py`: Скрипт обучения с Focal Loss и ArcFace
- `scripts/export_onnx.py`: Экспорт модели в ONNX
- `service/app.py`: FastAPI REST API сервис
- `integrations/ecoassist_postprocess.py`: Интеграция с EcoAssist

### Тестирование

```bash
# Тестирование API
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64>", "domain": "day"}'
```

## Известные ограничения

- Модель обучена на специфичных данных фотоловушек
- Требуется предобработка через MegaDetector
- Ночные кадры требуют отдельной модели или адаптации
- Частичные силуэты могут давать класс "unknown"

## Поддержка

Для вопросов и поддержки обращайтесь к документации проекта или создавайте issues в репозитории.

## Авторы

Проект разработан для интеграции в EcoAssist/AddaxAI пайплайн.

---

**Версия:** 1.0
**Дата:** 2025
