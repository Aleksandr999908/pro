#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Утилиты для работы с keypoints (ключевыми точками)
"""
import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime


def save_keypoints(
    image_id: str,
    keypoints: List[Dict],
    width: int,
    height: int,
    model: str = "fgc_day.onnx",
    timestamp: Optional[str] = None,
    output_dir: Union[str, Path] = "data/keypoints",
    objects: Optional[List[Dict]] = None
) -> Path:
    """
    Сохраняет keypoints для изображения/кадра в формате JSON

    Args:
        image_id: ID изображения (например, "video_001/000123.jpg")
        keypoints: Список keypoints, каждый с полями name, x, y, score
        width: Ширина изображения
        height: Высота изображения
        model: Название модели, использованной для детекции
        timestamp: Временная метка (ISO format), если None - текущее время
        output_dir: Директория для сохранения
        objects: Опциональный список объектов для многообъектных сцен

    Returns:
        Path к сохранённому файлу
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Формируем timestamp
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat() + "Z"

    # Формируем JSON структуру
    keypoints_data = {
        "image_id": image_id,
        "width": width,
        "height": height,
        "model": model,
        "keypoints": keypoints,
        "timestamp": timestamp
    }

    # Если есть объекты, добавляем их
    if objects is not None:
        keypoints_data["objects"] = objects

    # Определяем имя файла из image_id
    # Например: "video_001/000123.jpg" -> "video_001_000123.json"
    # Или просто используем image_id с заменой расширения
    if "/" in image_id:
        # Если есть путь, используем последнюю часть
        file_name = image_id.replace("/", "_").replace("\\", "_")
    else:
        file_name = image_id

    # Заменяем расширение на .json
    if "." in file_name:
        file_name = ".".join(file_name.split(".")[:-1]) + ".json"
    else:
        file_name = file_name + ".json"

    # Сохраняем JSON
    json_path = output_dir / file_name
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(keypoints_data, f, indent=2, ensure_ascii=False)

    return json_path


def save_keypoints_multi_object(
    image_id: str,
    objects: List[Dict],
    width: int,
    height: int,
    model: str = "fgc_day.onnx",
    timestamp: Optional[str] = None,
    output_dir: Union[str, Path] = "data/keypoints"
) -> Path:
    """
    Сохраняет keypoints для многообъектной сцены

    Args:
        image_id: ID изображения
        objects: Список объектов, каждый с полями:
                 - id: ID объекта
                 - class: Класс объекта (fox, wolf, etc.)
                 - bbox: Bounding box [x, y, w, h]
                 - keypoints: Список keypoints для объекта
        width: Ширина изображения
        height: Высота изображения
        model: Название модели
        timestamp: Временная метка
        output_dir: Директория для сохранения

    Returns:
        Path к сохранённому файлу
    """
    return save_keypoints(
        image_id=image_id,
        keypoints=[],  # Пустой список, т.к. keypoints в objects
        width=width,
        height=height,
        model=model,
        timestamp=timestamp,
        output_dir=output_dir,
        objects=objects
    )


def load_keypoints(
    image_id: str,
    keypoints_dir: Union[str, Path] = "data/keypoints"
) -> Optional[Dict]:
    """
    Загружает keypoints из JSON файла

    Args:
        image_id: ID изображения
        keypoints_dir: Директория с keypoints

    Returns:
        Словарь с данными keypoints или None, если файл не найден
    """
    keypoints_dir = Path(keypoints_dir)

    # Определяем имя файла
    if "/" in image_id:
        file_name = image_id.replace("/", "_").replace("\\", "_")
    else:
        file_name = image_id

    if "." in file_name:
        file_name = ".".join(file_name.split(".")[:-1]) + ".json"
    else:
        file_name = file_name + ".json"

    json_path = keypoints_dir / file_name

    if not json_path.exists():
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_keypoint(
    name: str,
    x: float,
    y: float,
    score: float
) -> Dict:
    """
    Создаёт структуру keypoint

    Args:
        name: Название точки (например, "nose", "left_ear")
        x: X координата
        y: Y координата
        score: Уверенность (0-1)

    Returns:
        Словарь с keypoint
    """
    return {
        "name": name,
        "x": float(x),
        "y": float(y),
        "score": float(score)
    }


def create_object(
    obj_id: int,
    obj_class: str,
    bbox: List[float],
    keypoints: List[Dict]
) -> Dict:
    """
    Создаёт структуру объекта для многообъектной сцены

    Args:
        obj_id: ID объекта
        obj_class: Класс объекта (fox, wolf, etc.)
        bbox: Bounding box [x, y, w, h]
        keypoints: Список keypoints для объекта

    Returns:
        Словарь с объектом
    """
    return {
        "id": int(obj_id),
        "class": obj_class,
        "bbox": [float(x) for x in bbox],
        "keypoints": keypoints
    }


