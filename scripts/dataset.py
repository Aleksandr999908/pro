#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Класс датасета для классификатора "Лиса vs Волк"
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from pathlib import Path
import json
import yaml
from sklearn.model_selection import train_test_split
from collections import Counter

class FoxWolfDataset(Dataset):
    """Датасет для классификации лиса/волк"""

    def __init__(self, data_dir, split='train', transform=None, config=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.config = config

        # Классы: 0 - fox, 1 - wolf, 2 - unknown
        self.class_names = ['fox', 'wolf', 'unknown']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        # Маппинг русских названий на английские
        self.russian_to_english = {
            'лисица': 'fox',
            'лисы': 'fox',
            'fox': 'fox',
            'волк': 'wolf',
            'волки': 'wolf',
            'wolf': 'wolf'
        }

        # Загружаем данные
        self.images, self.labels = self._load_data()

        # Разделение на train/val/test
        if split != 'all':
            self.images, self.labels = self._split_data(self.images, self.labels)

    def _load_data(self):
        """Загружает изображения и метки из директорий"""
        images = []
        labels = []

        # Структура: data/{frames|roi}/{month}/{class}/{video_id}/frame_*.jpg
        # Или просто: data/{frames|roi}/{month}/{class}/*.jpg
        roi_dir = self.data_dir
        if (roi_dir / 'roi').exists():
            roi_dir = roi_dir / 'roi'

        for month_dir in roi_dir.iterdir():
            if not month_dir.is_dir():
                continue

            for class_dir in month_dir.iterdir():
                if not class_dir.is_dir():
                    continue

                # Обрабатываем названия классов (поддержка русского и английского)
                class_name_orig = class_dir.name.lower()
                # Маппим русские названия на английские
                class_name = self.russian_to_english.get(class_name_orig, class_name_orig)

                if class_name not in self.class_to_idx:
                    class_name = 'unknown'

                label = self.class_to_idx[class_name]

                # Ищем все изображения в поддиректориях
                for img_path in class_dir.rglob('*.jpg'):
                    images.append(str(img_path))
                    labels.append(label)

                # Также проверяем поддиректории с video_id
                for video_dir in class_dir.iterdir():
                    if video_dir.is_dir():
                        for img_path in video_dir.glob('*.jpg'):
                            images.append(str(img_path))
                            labels.append(label)

        return images, labels

    def _split_data(self, images, labels):
        """Разделяет данные на train/val/test"""
        if self.config:
            train_split = self.config['data']['train_split']
            val_split = self.config['data']['val_split']
            test_split = self.config['data']['test_split']
        else:
            train_split, val_split, test_split = 0.7, 0.15, 0.15

        # Сначала разделяем train и temp (val+test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels, test_size=(1 - train_split),
            stratify=labels, random_state=42
        )

        # Затем разделяем temp на val и test
        test_size = test_split / (val_split + test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_size,
            stratify=y_temp, random_state=42
        )

        if self.split == 'train':
            return X_train, y_train
        elif self.split == 'val':
            return X_val, y_val
        elif self.split == 'test':
            return X_test, y_test
        else:
            return images, labels

    def get_class_counts(self):
        """Возвращает количество образцов по классам"""
        counts = Counter(self.labels)
        return [counts.get(i, 0) for i in range(len(self.class_names))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        try:
            # Загружаем изображение
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)

            # Применяем трансформации
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Базовая нормализация
                image = image.astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)

            return image, label
        except Exception as e:
            print(f"Ошибка при загрузке изображения {img_path}: {e}")
            # Возвращаем пустое изображение
            image = torch.zeros((3, 224, 224))
            return image, label
