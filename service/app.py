#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
REST API сервис для классификации "Лиса vs Волк"
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np
import cv2
from PIL import Image
import io
import base64
import onnxruntime as ort
import yaml
from pathlib import Path
from typing import Optional
import uvicorn

# Глобальные переменные для моделей
model_sessions = {}  # Словарь: domain -> model_session
config = None

class ClassificationRequest(BaseModel):
    """Запрос на классификацию"""
    image: str  # Base64 encoded image
    domain: Optional[str] = "day"  # day or night
    metadata: Optional[dict] = {}  # Дополнительные метаданные (month, location, etc.)

class ClassificationResponse(BaseModel):
    """Ответ с результатами классификации"""
    class_: str  # fox, wolf, unknown
    prob: float  # Вероятность предсказанного класса
    refine_conf: float  # Уверенность в уточнении
    probs: dict  # Вероятности для всех классов

def load_model(model_path: str, config_path: str, domain: str = "day"):
    """Загружает ONNX модель для указанного домена"""
    global model_sessions, config

    # Загружаем конфигурацию (только один раз)
    if config is None:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")

    # Создаем ONNX Runtime сессию
    providers = ['CPUExecutionProvider']
    if ort.get_device() == 'GPU':
        providers.insert(0, 'CUDAExecutionProvider')

    model_session = ort.InferenceSession(
        model_path,
        providers=providers
    )

    # Сохраняем сессию для домена
    model_sessions[domain] = model_session
    print(f"Модель загружена для домена '{domain}': {model_path}")


def get_model_session(domain: str = "day"):
    """Получает сессию модели для указанного домена, загружает если нужно"""
    global model_sessions, config

    # Если модель уже загружена, возвращаем её
    if domain in model_sessions:
        return model_sessions[domain]

    # Иначе загружаем модель
    if config is None:
        config_path = Path("models/config_fast.yaml")
        if not config_path.exists():
            config_path = Path("models/config.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    # Получаем путь к модели из конфигурации
    model_path = config.get('deployment', {}).get('models', {}).get(domain)
    if not model_path:
        # Fallback на старый формат
        if domain == "day":
            model_path = "models/fgc_day.onnx"
        elif domain == "night":
            model_path = "models/fgc_night.onnx"
        else:
            model_path = "models/fgc_day.onnx"

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}. Убедитесь, что файл существует.")

    config_path = Path("models/config_fast.yaml")
    if not config_path.exists():
        config_path = Path("models/config.yaml")

    # Загружаем модель
    try:
        load_model(str(model_path), str(config_path), domain)
    except Exception as e:
        raise RuntimeError(f"Ошибка загрузки модели {model_path}: {str(e)}")

    return model_sessions[domain]

def preprocess_image(image: Image.Image, input_size: int = 224) -> np.ndarray:
    """Предобрабатывает изображение для модели"""
    # Ресайз
    image = image.resize((input_size, input_size))

    # Конвертация в numpy array
    img_array = np.array(image, dtype=np.float32) / 255.0

    # Нормализация (ImageNet)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    # Перестановка размерностей: HWC -> CHW
    img_array = np.transpose(img_array, (2, 0, 1))

    # Добавляем batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Убеждаемся, что тип данных float32
    return img_array.astype(np.float32)

def decode_base64_image(image_str: str) -> Image.Image:
    """Декодирует Base64 строку в изображение"""
    try:
        # Убираем префикс data:image/...;base64, если есть
        if ',' in image_str:
            image_str = image_str.split(',')[1]

        image_bytes = base64.b64decode(image_str)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка декодирования изображения: {e}")

def classify_image(image: Image.Image, domain: str = "day") -> dict:
    """Классифицирует изображение"""
    global config

    # Получаем сессию модели для указанного домена
    model_session = get_model_session(domain)

    if model_session is None:
        raise HTTPException(status_code=500, detail=f"Модель не загружена для домена: {domain}")

    # Предобработка
    input_size = config.get('model', {}).get('input_size', 224)
    preprocessed = preprocess_image(image, input_size)

    # Убеждаемся, что тип данных float32
    preprocessed = preprocessed.astype(np.float32)

    # Инференс
    input_name = model_session.get_inputs()[0].name
    output_name = model_session.get_outputs()[0].name

    outputs = model_session.run([output_name], {input_name: preprocessed})
    logits = outputs[0][0]

    # Применяем softmax
    probs = softmax(logits)

    # Классы: 0 - fox, 1 - wolf, 2 - unknown
    class_names = ['fox', 'wolf', 'unknown']
    class_idx = np.argmax(probs)
    class_name = class_names[class_idx]
    prob = float(probs[class_idx])

    # Проверка порогов уверенности
    confidence_threshold = config.get('inference', {}).get('confidence_threshold', 0.7)
    class_gap_threshold = config.get('inference', {}).get('class_gap_threshold', 0.2)

    # Вычисляем зазор между двумя лучшими классами
    sorted_probs = np.sort(probs)[::-1]
    gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

    # Если уверенность низкая или зазор мал -> unknown
    if prob < confidence_threshold or gap < class_gap_threshold:
        class_name = 'unknown'
        prob = float(probs[2]) if len(probs) > 2 else 0.5

    return {
        'class': class_name,
        'prob': prob,
        'refine_conf': prob,
        'probs': {
            class_names[i]: float(probs[i]) for i in range(len(class_names))
        }
    }

def softmax(x):
    """Вычисляет softmax"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при запуске и завершении"""
    # Startup
    config_path = Path("models/config_fast.yaml")
    if not config_path.exists():
        config_path = Path("models/config.yaml")

    # Загружаем конфигурацию
    global config
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

    # Пытаемся загрузить модели для day и night
    day_model_path = Path("models/fgc_day.onnx")
    night_model_path = Path("models/fgc_night.onnx")

    if day_model_path.exists():
        try:
            load_model(str(day_model_path), str(config_path), "day")
        except Exception as e:
            print(f"Ошибка загрузки модели для day: {e}")
    else:
        print("ВНИМАНИЕ: Модель fgc_day.onnx не найдена. Будет загружена при первом запросе.")

    if night_model_path.exists():
        try:
            load_model(str(night_model_path), str(config_path), "night")
        except Exception as e:
            print(f"Ошибка загрузки модели для night: {e}")
    else:
        print("ВНИМАНИЕ: Модель fgc_night.onnx не найдена. Будет загружена при первом запросе.")

    yield

    # Shutdown (если нужно)

app = FastAPI(title="Fox/Wolf Classifier API", version="1.0.0", lifespan=lifespan)

# Подключаем статические файлы
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

@app.get("/")
async def root():
    """Главная страница - веб-интерфейс или API info"""
    # Если есть веб-интерфейс, возвращаем его
    index_path = static_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))

    # Иначе возвращаем JSON
    return {
        "service": "Fox/Wolf Classifier API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": {
            "day": "day" in model_sessions,
            "night": "night" in model_sessions
        }
    }

@app.get("/api")
async def api_info():
    """Информация об API"""
    return {
        "service": "Fox/Wolf Classifier API",
        "version": "1.0.0",
        "status": "running",
        "models_loaded": {
            "day": "day" in model_sessions,
            "night": "night" in model_sessions
        },
        "endpoints": {
            "health": "/health",
            "classify": "/classify",
            "classify_file": "/classify/file",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "models_loaded": {
            "day": "day" in model_sessions,
            "night": "night" in model_sessions
        }
    }

@app.post("/classify", response_model=ClassificationResponse)
async def classify(request: ClassificationRequest):
    """Классифицирует изображение"""
    try:
        # Декодируем изображение
        image = decode_base64_image(request.image)

        # Классификация
        domain = request.domain or "day"
        result = classify_image(image, domain)

        return ClassificationResponse(
            class_=result['class'],
            prob=result['prob'],
            refine_conf=result['refine_conf'],
            probs=result['probs']
        )
    except HTTPException:
        # Пробрасываем HTTPException как есть
        raise
    except Exception as e:
        # Логируем полную ошибку для отладки
        import traceback
        error_detail = f"Ошибка классификации: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_detail}")  # Выводим в консоль для отладки
        raise HTTPException(status_code=500, detail=f"Ошибка классификации: {str(e)}")

@app.post("/classify/file")
async def classify_file(file: UploadFile = File(...), domain: str = "day"):
    """Классифицирует изображение из загруженного файла"""
    try:
        # Читаем файл
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')

        # Классификация
        result = classify_image(image, domain)

        return JSONResponse(content=result)
    except HTTPException:
        # Пробрасываем HTTPException как есть
        raise
    except Exception as e:
        # Логируем полную ошибку для отладки
        import traceback
        error_detail = f"Ошибка классификации: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_detail}")  # Выводим в консоль для отладки
        raise HTTPException(status_code=500, detail=f"Ошибка классификации: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
