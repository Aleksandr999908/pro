#!/bin/bash
# Пример тестирования через curl

# Тест 1: Health check
echo "=== Проверка статуса API ==="
curl -s http://localhost:8000/health | python -m json.tool

echo -e "\n=== Тест классификации через файл ==="
# Тест 2: Классификация файла
curl -X POST "http://localhost:8000/classify/file" \
  -F "file=@data/frames/01.2025/Лисица/1/frame_000000.jpg" \
  -F "domain=day" | python -m json.tool










