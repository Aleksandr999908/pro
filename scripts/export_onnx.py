#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Скрипт для экспорта обученной модели в ONNX
"""
import torch
import torch.onnx
import timm
import yaml
import argparse
from pathlib import Path

def export_to_onnx(model_path, config_path, output_path, input_size=224):
    """Экспортирует PyTorch модель в ONNX"""

    # Загружаем конфигурацию
    with open(config_path, 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Загружаем модель
    device = torch.device('cpu')
    model = timm.create_model(
        config['model']['architecture'],
        pretrained=False,
        num_classes=config['model']['num_classes']
    )

    # Загружаем веса
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)

    # Создаем примерный вход
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

    # Экспорт в ONNX
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=config.get('export', {}).get('opset_version', 11),
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"Модель экспортирована в ONNX: {output_path}")

    # Оптимизация (если установлен onnxoptimizer)
    try:
        import onnx
        import onnxoptimizer

        onnx_model = onnx.load(str(output_path))
        optimized_model = onnxoptimizer.optimize(onnx_model)
        onnx.save(optimized_model, str(output_path))
        print("Модель оптимизирована")
    except ImportError:
        print("onnxoptimizer не установлен, пропускаем оптимизацию")

    return output_path

def main():
    parser = argparse.ArgumentParser(description='Экспорт модели в ONNX')
    parser.add_argument('--model', type=str, required=True,
                       help='Путь к PyTorch модели (.pth)')
    parser.add_argument('--config', type=str, default='models/config.yaml',
                       help='Путь к конфигурационному файлу')
    parser.add_argument('--output', type=str, required=True,
                       help='Путь для сохранения ONNX модели')
    parser.add_argument('--input_size', type=int, default=224,
                       help='Размер входного изображения')

    args = parser.parse_args()

    export_to_onnx(args.model, args.config, args.output, args.input_size)

if __name__ == "__main__":
    main()
