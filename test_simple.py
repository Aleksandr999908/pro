#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Простой пример тестирования одного изображения
"""
import requests
import base64
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Использование: python test_simple.py <путь_к_изображению>")
    sys.exit(1)

image_path = sys.argv[1]

if not Path(image_path).exists():
    print(f"Ошибка: файл не найден - {image_path}")
    sys.exit(1)

# Читаем и кодируем изображение
with open(image_path, 'rb') as f:
    img_base64 = base64.b64encode(f.read()).decode()

# Отправляем запрос
print(f"Отправляю запрос для: {image_path}")
response = requests.post(
    "http://localhost:8000/classify",
    json={"image": img_base64, "domain": "day"}
)

if response.status_code == 200:
    result = response.json()
    print(f"\nРезультат:")
    print(f"  Класс: {result['class_']}")
    print(f"  Уверенность: {result['prob']:.1%}")
    print(f"\nВсе вероятности:")
    for class_name, prob in result['probs'].items():
        print(f"  {class_name}: {prob:.1%}")
else:
    print(f"Ошибка: {response.status_code}")
    print(response.text)










