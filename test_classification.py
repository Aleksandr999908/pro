#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для тестирования классификации изображений
"""
import requests
import base64
import json
import sys
import os
from pathlib import Path

# Настройка кодировки для Windows
if sys.platform == 'win32':
    os.system('chcp 65001 > nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

def test_classification(image_path, expected_class=None):
    """Тестирует классификацию одного изображения"""
    print(f"\n{'='*60}")
    print(f"Тестирую: {image_path}")
    if expected_class:
        print(f"Ожидаемый класс: {expected_class}")
    print(f"{'='*60}")

    try:
        # Читаем изображение
        with open(image_path, 'rb') as f:
            img_base64 = base64.b64encode(f.read()).decode()

        # Отправляем запрос
        response = requests.post(
            "http://localhost:8000/classify",
            json={"image": img_base64, "domain": "day"},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Результат классификации:")
            print(f"   Класс: {result['class_']}")
            print(f"   Уверенность: {result['prob']:.2%}")
            print(f"   Вероятности:")
            for class_name, prob in result['probs'].items():
                bar = "#" * int(prob * 30)
                print(f"     {class_name:8s}: {prob:.2%} {bar}")

            # Проверяем, совпадает ли с ожидаемым
            if expected_class:
                if result['class_'] == expected_class.lower():
                    print(f"   [OK] Правильно распознано!")
                else:
                    print(f"   [WARN] Ожидался '{expected_class}', получен '{result['class_']}'")

            return result
        else:
            print(f"[ERROR] Ошибка: {response.status_code}")
            print(f"   {response.text}")
            return None

    except Exception as e:
        print(f"[ERROR] Ошибка при обработке: {e}")
        return None

if __name__ == "__main__":
    # Тестовые изображения
    test_images = [
        ("data/frames/01.2025/Волк/1/frame_000000.jpg", "wolf"),
        ("data/frames/01.2025/Волк/1/frame_000001.jpg", "wolf"),
        ("data/frames/01.2025/Лисица/1/frame_000000.jpg", "fox"),
        ("data/frames/01.2025/Лисица/1/frame_000001.jpg", "fox"),
    ]

    # Если передан аргумент - используем его как путь к изображению
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if Path(image_path).exists():
            test_classification(image_path)
        else:
            print(f"[ERROR] Файл не найден: {image_path}")
    else:
        # Тестируем все изображения из списка
        print("Тестирование классификации изображений")
        print(f"API: http://localhost:8000")

        results = []
        for image_path, expected_class in test_images:
            if Path(image_path).exists():
                result = test_classification(image_path, expected_class)
                if result:
                    results.append((image_path, expected_class, result['class_'], result['prob']))
            else:
                print(f"\n[WARN] Файл не найден: {image_path}")

        # Итоговая статистика
        if results:
            print(f"\n{'='*60}")
            print("Итоговая статистика:")
            print(f"{'='*60}")
            correct = sum(1 for _, exp, got, _ in results if exp == got)
            total = len(results)
            print(f"Правильно распознано: {correct}/{total} ({correct/total:.1%})")
            print(f"\nДетали:")
            for img_path, exp, got, prob in results:
                status = "[OK]" if exp == got else "[FAIL]"
                print(f"  {status} {Path(img_path).name}: ожидался '{exp}', получен '{got}' ({prob:.1%})")
