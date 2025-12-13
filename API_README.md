# 🚀 API Сервис - Инструкция по запуску

## Запуск API

```bash
python service/app.py
```

API будет доступен по адресу: **http://127.0.0.1:8000** или **http://localhost:8000**

## Эндпоинты

### 1. Главная страница
**GET** http://localhost:8000/

Возвращает информацию о сервисе

### 2. Проверка здоровья
**GET** http://localhost:8000/health

Проверяет, работает ли API и загружена ли модель

### 3. Классификация изображения (Base64)
**POST** http://localhost:8000/classify

**Тело запроса (JSON):**
```json
{
  "image": "<base64_encoded_image>",
  "domain": "day",  // или "night"
  "metadata": {
    "month": "2025-01",
    "location": "camera_1"
  }
}
```

**Ответ:**
```json
{
  "class": "fox",  // fox, wolf, или unknown
  "prob": 0.94,
  "refine_conf": 0.94,
  "probs": {
    "fox": 0.94,
    "wolf": 0.05,
    "unknown": 0.01
  }
}
```

### 4. Классификация изображения (файл)
**POST** http://localhost:8000/classify/file

**Запрос:** multipart/form-data
- `file`: изображение (jpg, png)
- `domain`: "day" или "night" (опционально)

**Ответ:** JSON с результатами классификации

## Интерактивная документация

После запуска API откройте в браузере:
- **http://localhost:8000/docs** - Swagger UI (интерактивная документация)
- **http://localhost:8000/redoc** - ReDoc (альтернативная документация)

## Примеры использования

### Python

```python
import requests
import base64

# Загрузите изображение
with open("test_image.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Отправьте запрос
response = requests.post(
    "http://localhost:8000/classify",
    json={
        "image": img_base64,
        "domain": "day"
    }
)

result = response.json()
print(f"Класс: {result['class']}, Уверенность: {result['prob']}")
```

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Классификация файла
curl -X POST http://localhost:8000/classify/file \
  -F "file=@test_image.jpg" \
  -F "domain=day"
```

### JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('domain', 'day');

fetch('http://localhost:8000/classify/file', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => console.log(data));
```

## Устранение проблем

### Порт 8000 занят

Измените порт в `service/app.py`:
```python
uvicorn.run(app, host="127.0.0.1", port=8001)
```

### Модель не загружена

Проверьте, что файл `models/fgc_day.onnx` существует.

Если модели нет, API все равно запустится, но будет возвращать ошибки при классификации.

### Ошибка кодировки

Убедитесь, что файлы конфигурации сохранены в UTF-8.

## Использование с Docker

```bash
docker build -t fox-wolf-classifier -f service/Dockerfile .
docker run -p 8000:8000 -v $(pwd)/models:/app/models fox-wolf-classifier
```

Или через docker-compose:
```bash
docker-compose up
```
