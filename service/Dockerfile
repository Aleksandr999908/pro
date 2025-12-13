# Dockerfile для REST API сервиса классификации
FROM python:3.10-slim

WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код сервиса
COPY service/ ./service/
COPY models/ ./models/

# Создаем директорию для моделей если её нет
RUN mkdir -p models

# Экспортируем порт
EXPOSE 8000

# Запускаем приложение
CMD ["python", "-m", "uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
