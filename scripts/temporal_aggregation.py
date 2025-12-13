#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Утилита для временной агрегации результатов классификации по видеокадрам
"""
import json
import numpy as np
from collections import Counter
from typing import List, Dict
import argparse

def majority_vote(classifications: List[Dict], window_size: int = 5) -> List[Dict]:
    """
    Применяет majority vote для временной агрегации

    Args:
        classifications: Список результатов классификации по кадрам
        window_size: Размер окна для агрегации

    Returns:
        Список результатов с примененной агрегацией
    """
    if len(classifications) < window_size:
        return classifications

    refined = []

    for i in range(len(classifications)):
        # Создаем окно вокруг текущего кадра
        start = max(0, i - window_size // 2)
        end = min(len(classifications), i + window_size // 2 + 1)
        window = classifications[start:end]

        # Собираем классы и уверенности
        classes = [item.get('class', 'unknown') for item in window]
        confidences = [item.get('conf', 0.0) for item in window]

        # Majority vote
        class_counts = Counter(classes)
        if class_counts:
            most_common_class, count = class_counts.most_common(1)[0]
            avg_conf = np.mean(confidences)

            # Обновляем результат
            result = classifications[i].copy()
            result['class'] = most_common_class
            result['conf'] = float(avg_conf)
            result['vote_count'] = count
            result['window_size'] = len(window)
        else:
            result = classifications[i].copy()

        refined.append(result)

    return refined

def median_aggregation(classifications: List[Dict], window_size: int = 5) -> List[Dict]:
    """
    Применяет медианную агрегацию для временной консистентности

    Args:
        classifications: Список результатов классификации по кадрам
        window_size: Размер окна для агрегации

    Returns:
        Список результатов с примененной агрегацией
    """
    if len(classifications) < window_size:
        return classifications

    # Для медианной агрегации нужны численные представления классов
    class_to_num = {'fox': 0, 'wolf': 1, 'unknown': 2}
    num_to_class = {0: 'fox', 1: 'wolf', 2: 'unknown'}

    refined = []

    for i in range(len(classifications)):
        start = max(0, i - window_size // 2)
        end = min(len(classifications), i + window_size // 2 + 1)
        window = classifications[start:end]

        # Преобразуем классы в числа
        class_nums = [class_to_num.get(item.get('class', 'unknown'), 2) for item in window]
        confidences = [item.get('conf', 0.0) for item in window]

        # Медиана классов и медиана уверенности
        median_class_num = int(np.median(class_nums))
        median_conf = float(np.median(confidences))

        result = classifications[i].copy()
        result['class'] = num_to_class[median_class_num]
        result['conf'] = median_conf
        result['method'] = 'median'

        refined.append(result)

    return refined

def temporal_pooling(classifications: List[Dict], window_size: int = 5,
                     pool_type: str = 'max') -> List[Dict]:
    """
    Применяет временное пулирование для агрегации

    Args:
        classifications: Список результатов классификации по кадрам
        window_size: Размер окна для агрегации
        pool_type: Тип пулирования (max, mean, avg)

    Returns:
        Список результатов с примененной агрегацией
    """
    if len(classifications) < window_size:
        return classifications

    refined = []

    for i in range(len(classifications)):
        start = max(0, i - window_size // 2)
        end = min(len(classifications), i + window_size // 2 + 1)
        window = classifications[start:end]

        # Собираем вероятности для каждого класса
        class_probs = {'fox': [], 'wolf': [], 'unknown': []}

        for item in window:
            probs = item.get('probs', {})
            for class_name in class_probs.keys():
                class_probs[class_name].append(probs.get(class_name, 0.0))

        # Применяем пулирование
        if pool_type == 'max':
            pooled_probs = {k: max(v) if v else 0.0 for k, v in class_probs.items()}
        elif pool_type == 'mean' or pool_type == 'avg':
            pooled_probs = {k: np.mean(v) if v else 0.0 for k, v in class_probs.items()}
        else:
            pooled_probs = {k: np.mean(v) if v else 0.0 for k, v in class_probs.items()}

        # Нормализуем вероятности
        total = sum(pooled_probs.values())
        if total > 0:
            pooled_probs = {k: v / total for k, v in pooled_probs.items()}

        # Определяем класс
        best_class = max(pooled_probs, key=pooled_probs.get)
        best_conf = pooled_probs[best_class]

        result = classifications[i].copy()
        result['class'] = best_class
        result['conf'] = float(best_conf)
        result['probs'] = {k: float(v) for k, v in pooled_probs.items()}
        result['method'] = f'temporal_{pool_type}'

        refined.append(result)

    return refined

def main():
    parser = argparse.ArgumentParser(
        description='Временная агрегация результатов классификации'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Путь к JSON файлу с результатами классификации')
    parser.add_argument('--output', type=str, required=True,
                       help='Путь для сохранения агрегированных результатов')
    parser.add_argument('--method', type=str, default='majority_vote',
                       choices=['majority_vote', 'median', 'temporal_max', 'temporal_mean'],
                       help='Метод агрегации')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Размер окна для агрегации')

    args = parser.parse_args()

    # Загружаем результаты
    with open(args.input, 'r') as f:
        data = json.load(f)
        if isinstance(data, list):
            classifications = data
        else:
            classifications = data.get('classifications', [])

    # Применяем агрегацию
    if args.method == 'majority_vote':
        refined = majority_vote(classifications, args.window_size)
    elif args.method == 'median':
        refined = median_aggregation(classifications, args.window_size)
    elif args.method == 'temporal_max':
        refined = temporal_pooling(classifications, args.window_size, 'max')
    elif args.method == 'temporal_mean':
        refined = temporal_pooling(classifications, args.window_size, 'mean')
    else:
        refined = majority_vote(classifications, args.window_size)

    # Сохраняем результаты
    with open(args.output, 'w') as f:
        json.dump(refined, f, indent=2)

    print(f"Агрегация завершена. Результаты сохранены в {args.output}")
    print(f"Метод: {args.method}, Размер окна: {args.window_size}")
    print(f"Обработано кадров: {len(refined)}")

if __name__ == "__main__":
    main()
