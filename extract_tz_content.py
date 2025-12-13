#!/usr/bin/env python
# -*- coding: utf-8 -*-
import zipfile
import xml.etree.ElementTree as ET
import os

def read_docx(filepath):
    """Читает текст из .docx файла"""
    text_parts = []
    try:
        with zipfile.ZipFile(filepath, 'r') as z:
            if 'word/document.xml' in z.namelist():
                xml = z.read('word/document.xml')
                root = ET.fromstring(xml)
                ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

                for p in root.findall('.//w:p', ns):
                    para_text = []
                    for t in p.findall('.//w:t', ns):
                        if t.text:
                            para_text.append(t.text)
                    if para_text:
                        text_parts.append(''.join(para_text))
    except Exception as e:
        return f"ERROR: {e}"
    return '\n'.join(text_parts)

# Читаем оба файла
files = {
    'instruction': 'FGC_EcoAssist_Единая_инструкция.docx',
    'recommendations': 'Рекомендации_EcoAssist_Лисы_Волки.docx'
}

results = {}
for key, filepath in files.items():
    if os.path.exists(filepath):
        results[key] = read_docx(filepath)
        print(f"Прочитан: {filepath} ({len(results[key])} символов)")
    else:
        results[key] = f"ФАЙЛ НЕ НАЙДЕН: {filepath}"

# Сохраняем в файл
output = f"""
{'='*80}
ТЕХНИЧЕСКОЕ ЗАДАНИЕ ПРОЕКТА
{'='*80}

ФАЙЛ 1: Единая инструкция
{'='*80}
{results.get('instruction', '')}

{'='*80}
ФАЙЛ 2: Рекомендации
{'='*80}
{results.get('recommendations', '')}
"""

with open('TZ_CONTENT.txt', 'w', encoding='utf-8') as f:
    f.write(output)

print("\nСодержимое сохранено в TZ_CONTENT.txt")
