#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для извлечения кадров из видеоархива
Извлекает кадры с частотой 1 fps и сохраняет с ROI из MegaDetector
"""
import os
import cv2
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import imageio

def extract_frames_from_video(video_path, output_dir, fps=1.0, preserve_structure=True):
    """Извлекает кадры из видео с заданной частотой"""
    video_name = Path(video_path).stem

    if preserve_structure:
        # Сохраняем структуру: месяц/класс/видео/кадры
        # Определяем путь относительно video_dir
        # Например: data/videos/лисы-волки/01.2025/Волк/1.MP4
        parts = video_path.parts
        if 'лисы-волки' in parts:
            idx = parts.index('лисы-волки')
            if idx + 3 < len(parts):
                month = parts[idx + 1]
                class_name = parts[idx + 2]
                frame_dir = Path(output_dir) / month / class_name / video_name
            else:
                frame_dir = Path(output_dir) / video_name
        else:
            frame_dir = Path(output_dir) / video_name
    else:
        frame_dir = Path(output_dir) / video_name

    frame_dir.mkdir(parents=True, exist_ok=True)

    try:
        reader = imageio.get_reader(video_path, 'ffmpeg')
        fps_video = reader.get_meta_data().get('fps', 30)  # По умолчанию 30 fps
        if fps_video <= 0:
            fps_video = 30
        frame_interval = max(1, int(fps_video / fps))

        frame_count = 0
        saved_count = 0

        for i, frame in enumerate(reader):
            if i % frame_interval == 0:
                frame_path = frame_dir / f"frame_{saved_count:06d}.jpg"
                imageio.imwrite(frame_path, frame)
                saved_count += 1
            frame_count += 1

        reader.close()
        print(f"Извлечено {saved_count} кадров из {video_path}")
        return saved_count
    except Exception as e:
        print(f"Ошибка при обработке {video_path}: {e}")
        return 0

def copy_image_file(img_path, output_dir, preserve_structure=True):
    """Копирует JPG файл в структуру output_dir"""
    from shutil import copy2

    img_name = Path(img_path).name

    if preserve_structure:
        # Сохраняем структуру: месяц/класс/видео/изображение
        parts = img_path.parts
        if 'лисы-волки' in parts:
            idx = parts.index('лисы-волки')
            if idx + 3 < len(parts):
                month = parts[idx + 1]
                class_name = parts[idx + 2]
                output_path = Path(output_dir) / month / class_name / img_name
            else:
                output_path = Path(output_dir) / img_name
        else:
            output_path = Path(output_dir) / img_name
    else:
        output_path = Path(output_dir) / img_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    copy2(img_path, output_path)
    return 1

def process_megadetector_roi(frame_path, megadetector_json, padding=0.15):
    """Извлекает ROI из кадра на основе результатов MegaDetector"""
    if not os.path.exists(megadetector_json):
        return None

    with open(megadetector_json, 'r') as f:
        md_results = json.load(f)

    frame_name = Path(frame_path).name

    # Ищем соответствующий результат в JSON
    for detection in md_results.get('images', []):
        if detection.get('file') == frame_name:
            detections = detection.get('detections', [])

            # Ищем детекции "animal"
            for det in detections:
                if det.get('category') == '1':  # 1 = animal в MegaDetector
                    conf = det.get('conf', 0)
                    bbox = det.get('bbox', [])

                    if conf > 0.5:  # Фильтр уверенности
                        # Добавляем паддинг
                        x, y, w, h = bbox
                        img = cv2.imread(str(frame_path))
                        if img is None:
                            continue

                        img_h, img_w = img.shape[:2]

                        # Вычисляем ROI с паддингом
                        pad_x = int(w * padding)
                        pad_y = int(h * padding)

                        x1 = max(0, int(x - pad_x))
                        y1 = max(0, int(y - pad_y))
                        x2 = min(img_w, int(x + w + pad_x))
                        y2 = min(img_h, int(y + h + pad_y))

                        roi = img[y1:y2, x1:x2]
                        return roi, (x1, y1, x2, y2)

    return None

def main():
    parser = argparse.ArgumentParser(description='Извлечение кадров из видео')
    parser.add_argument('--video_dir', type=str, default='data/videos',
                       help='Директория с видео файлами')
    parser.add_argument('--output_dir', type=str, default='data/frames',
                       help='Директория для сохранения кадров')
    parser.add_argument('--fps', type=float, default=1.0,
                       help='Частота извлечения кадров (кадров/сек)')
    parser.add_argument('--megadetector_dir', type=str, default=None,
                       help='Директория с JSON результатами MegaDetector')
    parser.add_argument('--roi_output', type=str, default='data/roi',
                       help='Директория для сохранения ROI кадров')
    parser.add_argument('--preserve_structure', action='store_true', default=True,
                       help='Сохранять структуру папок (месяц/класс)')
    parser.add_argument('--process_images', action='store_true', default=True,
                       help='Обрабатывать JPG файлы как готовые кадры')

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ищем все видео файлы
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.rglob(f'*{ext}'))

    # Ищем изображения, если включена обработка
    image_files = []
    if args.process_images:
        image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
        for ext in image_extensions:
            image_files.extend(video_dir.rglob(f'*{ext}'))

    print(f"Найдено {len(video_files)} видео файлов")
    if image_files:
        print(f"Найдено {len(image_files)} изображений")

    total_frames = 0

    # Обрабатываем видео
    for video_path in tqdm(video_files, desc="Обработка видео"):
        frames_extracted = extract_frames_from_video(
            video_path, output_dir, args.fps, args.preserve_structure
        )
        total_frames += frames_extracted

    # Обрабатываем изображения
    if image_files:
        for img_path in tqdm(image_files, desc="Обработка изображений"):
            copied = copy_image_file(img_path, output_dir, args.preserve_structure)
            total_frames += copied

    # Если указана директория MegaDetector, обрабатываем ROI
    if args.megadetector_dir and args.roi_output:
        roi_output_path = Path(args.roi_output)
        roi_output_path.mkdir(parents=True, exist_ok=True)

        # Обрабатываем все кадры для извлечения ROI
        print("\nИзвлечение ROI из кадров...")
        frame_files = list(output_dir.rglob("*.jpg"))

        for frame_path in tqdm(frame_files, desc="Извлечение ROI"):
            # Определяем структуру для ROI
            relative_path = frame_path.relative_to(output_dir)
            roi_path = roi_output_path / relative_path
            roi_path.parent.mkdir(parents=True, exist_ok=True)

            # Пока просто копируем кадры (ROI будет извлечено после MegaDetector)
            # Или можно использовать весь кадр как ROI
            from shutil import copy2
            copy2(frame_path, roi_path)

    print(f"\nВсего обработано: {total_frames} кадров")
    print(f"Кадры сохранены в: {output_dir}")
    if args.megadetector_dir and args.roi_output:
        print(f"ROI кадры сохранены в: {args.roi_output}")

if __name__ == "__main__":
    main()
