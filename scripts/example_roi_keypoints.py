#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Пример использования утилит для работы с ROI и keypoints
"""
import sys
from pathlib import Path

# Добавляем путь к скриптам
sys.path.insert(0, str(Path(__file__).parent))

from roi_utils import (
    save_roi_mask,
    save_roi_polygons,
    bbox_to_polygon,
    create_mask_from_polygons,
    extract_roi_from_megadetector
)
from keypoints_utils import (
    save_keypoints,
    save_keypoints_multi_object,
    create_keypoint,
    create_object
)
import cv2
import numpy as np


def example_roi_mask():
    """Пример сохранения ROI-маски"""
    print("Пример 1: Сохранение ROI-маски")

    # Создаём тестовую маску (чёрно-белое изображение)
    mask = np.zeros((720, 1280), dtype=np.uint8)
    # Рисуем белый прямоугольник (ROI область)
    cv2.rectangle(mask, (100, 100), (500, 400), 255, -1)

    # Сохраняем маску
    mask_path = save_roi_mask(
        mask=mask,
        video_id="video_001",
        frame_idx=123,
        output_dir="data/roi"
    )
    print(f"Маска сохранена: {mask_path}")


def example_roi_polygons():
    """Пример сохранения ROI-координат в формате JSON"""
    print("\nПример 2: Сохранение ROI-координат (JSON)")

    # Создаём полигоны (например, из bounding box)
    bbox1 = [100, 100, 400, 300]  # [x, y, w, h]
    polygon1 = bbox_to_polygon(bbox1)

    bbox2 = [600, 200, 300, 250]
    polygon2 = bbox_to_polygon(bbox2)

    polygons = [polygon1, polygon2]

    # Сохраняем JSON
    json_path = save_roi_polygons(
        polygons=polygons,
        video_id="video_001",
        frame_idx=123,
        output_dir="data/roi"
    )
    print(f"JSON сохранён: {json_path}")


def example_keypoints_single():
    """Пример сохранения keypoints для одного объекта"""
    print("\nПример 3: Сохранение keypoints (один объект)")

    # Создаём keypoints
    keypoints = [
        create_keypoint("nose", 512.3, 340.8, 0.94),
        create_keypoint("left_ear", 480.1, 310.7, 0.89),
        create_keypoint("right_ear", 544.5, 310.2, 0.91),
        create_keypoint("tail_base", 600.0, 450.0, 0.85)
    ]

    # Сохраняем keypoints
    json_path = save_keypoints(
        image_id="video_001/000123.jpg",
        keypoints=keypoints,
        width=1280,
        height=720,
        model="fgc_day.onnx"
    )
    print(f"Keypoints сохранены: {json_path}")


def example_keypoints_multi_object():
    """Пример сохранения keypoints для многообъектной сцены"""
    print("\nПример 4: Сохранение keypoints (многообъектная сцена)")

    # Создаём объекты
    objects = [
        create_object(
            obj_id=1,
            obj_class="fox",
            bbox=[100, 100, 200, 250],
            keypoints=[
                create_keypoint("nose", 180.0, 200.0, 0.92),
                create_keypoint("left_ear", 150.0, 120.0, 0.88),
                create_keypoint("right_ear", 210.0, 120.0, 0.90)
            ]
        ),
        create_object(
            obj_id=2,
            obj_class="wolf",
            bbox=[500, 150, 300, 350],
            keypoints=[
                create_keypoint("nose", 620.0, 280.0, 0.95),
                create_keypoint("left_ear", 550.0, 180.0, 0.91),
                create_keypoint("right_ear", 690.0, 180.0, 0.93)
            ]
        )
    ]

    # Сохраняем keypoints для многообъектной сцены
    json_path = save_keypoints_multi_object(
        image_id="video_001/000123.jpg",
        objects=objects,
        width=1280,
        height=720,
        model="fgc_day.onnx"
    )
    print(f"Keypoints (многообъектные) сохранены: {json_path}")


def example_roi_from_megadetector():
    """Пример извлечения ROI из результатов MegaDetector"""
    print("\nПример 5: Извлечение ROI из MegaDetector")

    # Пример использования (требует реальные файлы)
    # frame_path = Path("data/frames/video_001/frame_000123.jpg")
    # megadetector_json = Path("data/megadetector_results.json")
    #
    # if frame_path.exists() and megadetector_json.exists():
    #     # Сохраняем в формате JSON
    #     json_path = extract_roi_from_megadetector(
    #         frame_path=frame_path,
    #         megadetector_json=megadetector_json,
    #         video_id="video_001",
    #         frame_idx=123,
    #         format="json"
    #     )
    #     print(f"ROI из MegaDetector сохранён (JSON): {json_path}")
    #
    #     # Или в формате маски
    #     mask_path = extract_roi_from_megadetector(
    #         frame_path=frame_path,
    #         megadetector_json=megadetector_json,
    #         video_id="video_001",
    #         frame_idx=123,
    #         format="mask"
    #     )
    #     print(f"ROI из MegaDetector сохранён (маска): {mask_path}")

    print("(Требует реальные файлы - закомментировано)")


if __name__ == "__main__":
    print("Примеры использования утилит ROI и keypoints\n")
    print("=" * 60)

    # Создаём директории
    Path("data/roi").mkdir(parents=True, exist_ok=True)
    Path("data/keypoints").mkdir(parents=True, exist_ok=True)

    # Запускаем примеры
    example_roi_mask()
    example_roi_polygons()
    example_keypoints_single()
    example_keypoints_multi_object()
    example_roi_from_megadetector()

    print("\n" + "=" * 60)
    print("Примеры выполнены!")


