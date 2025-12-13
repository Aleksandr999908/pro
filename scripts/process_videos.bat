@echo off
REM Скрипт для обработки всех видео в папке data/videos (Windows)

echo Начало обработки видео...
echo ==========================

REM Извлечение кадров из видео
python scripts/extract_frames.py --video_dir data/videos --output_dir data/frames --fps 1.0 --roi_output data/roi --preserve_structure --process_images

echo.
echo Обработка завершена!
echo Кадры сохранены в: data/frames
echo ROI кадры сохранены в: data/roi
pause
