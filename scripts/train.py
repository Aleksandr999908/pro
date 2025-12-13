#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт обучения классификатора "Лиса vs Волк"
Использует EfficientNet/MobileNet с Focal Loss или ArcFace Loss
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
import yaml
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from dataset import FoxWolfDataset

class FocalLoss(nn.Module):
    """Focal Loss для решения проблемы несбалансированных классов"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class ArcFaceLoss(nn.Module):
    """ArcFace Loss для метрического обучения"""
    def __init__(self, num_classes, embedding_size=512, m=0.5, s=64.0):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.m = m
        self.s = s
        self.embedding_size = embedding_size
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Нормализация
        embeddings = nn.functional.normalize(embeddings, dim=1)
        weight = nn.functional.normalize(self.weight, dim=1)

        # Косинус угла
        cosine = torch.mm(embeddings, weight.t())

        # ArcFace: добавляем маржу
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_theta = theta[torch.arange(len(labels)), labels].view(-1, 1)
        theta_with_margin = target_theta + self.m
        cosine_m = torch.cos(theta_with_margin)

        # Заменяем только для правильных классов
        output = cosine.clone()
        output[torch.arange(len(labels)), labels] = cosine_m.squeeze()
        output *= self.s

        return nn.CrossEntropyLoss()(output, labels)

def get_transforms(config, domain='day'):
    """Получает трансформации для аугментации"""
    if domain == 'day':
        augs = config['data']['augmentations']['day']
    else:
        augs = config['data']['augmentations']['night']

    transform_list = []

    if 'horizontal_flip' in augs:
        transform_list.append(A.HorizontalFlip(p=0.5))
    if 'color_jitter' in augs:
        transform_list.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5))
    if 'random_crop' in augs:
        transform_list.append(A.RandomCrop(config['model']['input_size'], config['model']['input_size']))
    if 'clahe' in augs:
        transform_list.append(A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5))

    # Всегда ресайз
    transform_list.append(A.Resize(config['model']['input_size'], config['model']['input_size']))
    transform_list.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transform_list.append(ToTensorV2())

    return A.Compose(transform_list)

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, f1

def validate(model, dataloader, criterion, device):
    """Валидация модели"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    # AUROC (для бинарной классификации нужно адаптировать)
    try:
        if len(np.unique(all_labels)) > 2:
            labels_bin = label_binarize(all_labels, classes=[0, 1, 2])
            auroc = roc_auc_score(labels_bin, all_probs, average='macro', multi_class='ovr')
        else:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
    except:
        auroc = 0.0

    return epoch_loss, f1, balanced_acc, auroc

def main():
    parser = argparse.ArgumentParser(description='Обучение классификатора Лиса/Волк')
    parser.add_argument('--config', type=str, default='models/config.yaml',
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Директория с данными')
    parser.add_argument('--output_dir', type=str, default='models/checkpoints',
                       help='Директория для сохранения чекпоинтов')

    args = parser.parse_args()

    # Загружаем конфигурацию
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")

    # Создаем датасет
    train_dataset = FoxWolfDataset(
        args.data_dir, split='train',
        transform=get_transforms(config, 'day'),
        config=config
    )

    val_dataset = FoxWolfDataset(
        args.data_dir, split='val',
        transform=get_transforms(config, 'day'),
        config=config
    )

    # Взвешенный сэмплинг для балансировки классов
    class_counts = train_dataset.get_class_counts()
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = [class_weights[label] for label in train_dataset.labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'],
        sampler=sampler, num_workers=4
    )

    val_loader = DataLoader(
        val_dataset, batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=4
    )

    # Создаем модель
    model_name = config['model']['architecture']
    model = timm.create_model(
        model_name,
        pretrained=config['model']['pretrained'],
        num_classes=config['model']['num_classes']
    )
    model = model.to(device)

    # Функция потерь
    loss_type = config['training']['loss']
    if loss_type == 'focal':
        criterion = FocalLoss(
            alpha=config['training']['focal_alpha'],
            gamma=config['training']['focal_gamma']
        )
    elif loss_type == 'arcface':
        # Для ArcFace нужна специальная архитектура с эмбеддингами
        # Упрощенная версия - используем Focal Loss
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # Оптимизатор
    optimizer = optim.SGD(
        model.parameters(),
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['training']['num_epochs']
    )

    # Обучение
    best_f1 = 0.0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config['training']['num_epochs']):
        print(f"\nЭпоха {epoch+1}/{config['training']['num_epochs']}")

        train_loss, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_f1, val_bal_acc, val_auroc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val Balanced Acc: {val_bal_acc:.4f}, Val AUROC: {val_auroc:.4f}")

        # Сохраняем лучшую модель
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': val_f1,
            }, output_dir / 'best_model.pth')
            print(f"Сохранина лучшая модель с F1: {val_f1:.4f}")

    print(f"\nОбучение завершено. Лучший F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()
