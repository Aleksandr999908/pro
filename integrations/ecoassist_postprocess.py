#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Интеграция классификатора "Лиса vs Волк" в EcoAssist/AddaxAI
Постпроцессор для уточнения классификации после MegaDetector
"""
import json
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import cv2
from PIL import Image
import base64
import io

class FoxWolfRefiner:
    """Класс для уточнения классификации лиса/волк"""

    def __init__(self, api_url: str = "http://localhost:8000",
                 confidence_threshold: float = 0.7,
                 class_gap_threshold: float = 0.2):
        """
        Инициализация рефайнера

        Args:
            api_url: URL REST API сервиса классификации
            confidence_threshold: Порог уверенности для принятия решения
            class_gap_threshold: Минимальный зазор между классами
        """
        self.api_url = api_url
        self.confidence_threshold = confidence_threshold
        self.class_gap_threshold = class_gap_threshold

    def extract_roi(self, image_path: str, bbox: List[float], padding: float = 0.15) -> Image.Image:
        """
        Извлекает ROI из изображения на основе bounding box

        Args:
            image_path: Путь к изображению
            bbox: Bounding box [x, y, width, height]
            padding: Дополнительный паддинг вокруг ROI

        Returns:
            PIL Image с ROI
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        h, w = image.shape[:2]
        x, y, width, height = bbox

        # Добавляем паддинг
        pad_x = int(width * padding)
        pad_y = int(height * padding)

        x1 = max(0, int(x - pad_x))
        y1 = max(0, int(y - pad_y))
        x2 = min(w, int(x + width + pad_x))
        y2 = min(h, int(y + height + pad_y))

        roi = image[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_pil = Image.fromarray(roi_rgb)

        return roi_pil

    def image_to_base64(self, image: Image.Image) -> str:
        """Конвертирует PIL Image в Base64 строку"""
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str

    def classify(self, image: Image.Image, domain: str = "day") -> Dict:
        """
        Классифицирует изображение через REST API

        Args:
            image: PIL Image
            domain: Домен (day/night)

        Returns:
            Словарь с результатами классификации
        """
        # Конвертируем в base64
        img_base64 = self.image_to_base64(image)

        # Отправляем запрос
        payload = {
            "image": img_base64,
            "domain": domain,
            "metadata": {}
        }

        try:
            response = requests.post(
                f"{self.api_url}/classify",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к API: {e}")
            return {
                "class": "unknown",
                "prob": 0.0,
                "refine_conf": 0.0,
                "probs": {"fox": 0.0, "wolf": 0.0, "unknown": 1.0}
            }

    def process_megadetector_result(self, megadetector_json: Dict,
                                    image_path: str,
                                    domain: str = "day") -> Dict:
        """
        Обрабатывает результат MegaDetector и добавляет уточнение

        Args:
            megadetector_json: JSON результат MegaDetector
            image_path: Путь к изображению
            domain: Домен (day/night)

        Returns:
            Обновленный JSON с полями refine_species и refine_conf
        """
        result = megadetector_json.copy()

        # Ищем детекции животных
        detections = result.get('detections', [])
        if not detections:
            return result

        # Обрабатываем каждую детекцию животного
        refined_detections = []
        for det in detections:
            if det.get('category') == '1':  # 1 = animal в MegaDetector
                conf = det.get('conf', 0)

                # Обрабатываем только уверенные детекции
                if conf > 0.5:
                    bbox = det.get('bbox', [])

                    # Извлекаем ROI
                    try:
                        roi = self.extract_roi(image_path, bbox)

                        # Классифицируем
                        classification = self.classify(roi, domain)

                        # Добавляем уточнение
                        det['refine_species'] = classification['class']
                        det['refine_conf'] = classification['refine_conf']
                        det['refine_probs'] = classification['probs']
                    except Exception as e:
                        print(f"Ошибка при обработке детекции: {e}")
                        det['refine_species'] = 'unknown'
                        det['refine_conf'] = 0.0

            refined_detections.append(det)

        result['detections'] = refined_detections
        return result

    def process_video_frames(self, frames: List[Dict],
                            temporal_window: int = 5) -> List[Dict]:
        """
        Применяет временную агрегацию для видео кадров

        Args:
            frames: Список результатов классификации по кадрам
            temporal_window: Размер окна для агрегации

        Returns:
            Список результатов с примененной агрегацией
        """
        if len(frames) < temporal_window:
            return frames

        refined_frames = []

        for i in range(len(frames)):
            # Создаем окно вокруг текущего кадра
            start = max(0, i - temporal_window // 2)
            end = min(len(frames), i + temporal_window // 2 + 1)
            window = frames[start:end]

            # Majority vote
            classes = [f.get('refine_species', 'unknown') for f in window]
            confidences = [f.get('refine_conf', 0.0) for f in window]

            # Находим наиболее частый класс
            from collections import Counter
            class_counts = Counter(classes)
            if class_counts:
                most_common = class_counts.most_common(1)[0][0]
                avg_conf = np.mean(confidences)

                frames[i]['refine_species'] = most_common
                frames[i]['refine_conf'] = float(avg_conf)

            refined_frames.append(frames[i])

        return refined_frames

def process_ecoassist_result(input_json_path: str,
                             output_json_path: str,
                             image_path: str,
                             api_url: str = "http://localhost:8000",
                             domain: str = "day"):
    """
    Основная функция для обработки результата EcoAssist

    Args:
        input_json_path: Путь к JSON результату MegaDetector/EcoAssist
        output_json_path: Путь для сохранения обновленного JSON
        image_path: Путь к изображению
        api_url: URL REST API сервиса
        domain: Домен (day/night)
    """
    # Загружаем JSON
    with open(input_json_path, 'r') as f:
        result = json.load(f)

    # Создаем рефайнер
    refiner = FoxWolfRefiner(api_url=api_url)

    # Обрабатываем результат
    refined_result = refiner.process_megadetector_result(
        result, image_path, domain
    )

    # Сохраняем обновленный JSON
    with open(output_json_path, 'w') as f:
        json.dump(refined_result, f, indent=2)

    print(f"Результат сохранен: {output_json_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Постпроцессор для уточнения классификации в EcoAssist'
    )
    parser.add_argument('--input_json', type=str, required=True,
                       help='Путь к JSON результату MegaDetector')
    parser.add_argument('--output_json', type=str, required=True,
                       help='Путь для сохранения обновленного JSON')
    parser.add_argument('--image', type=str, required=True,
                       help='Путь к изображению')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000',
                       help='URL REST API сервиса')
    parser.add_argument('--domain', type=str, default='day',
                       choices=['day', 'night'],
                       help='Домен изображения (day/night)')

    args = parser.parse_args()

    process_ecoassist_result(
        args.input_json,
        args.output_json,
        args.image,
        args.api_url,
        args.domain
    )
