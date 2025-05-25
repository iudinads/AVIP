from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

font_path = "Noto_Sans_Ugaritic/NotoSansUgaritic-Regular.ttf"
default_font_size = 52
alphabet_chars = [
    "𐎀", "𐎁", "𐎂", "𐎃", "𐎄", "𐎅", "𐎆", "𐎇", "𐎈", "𐎉",
    "𐎊", "𐎋", "𐎌", "𐎍", "𐎎", "𐎏", "𐎐", "𐎑", "𐎒", "𐎓",
    "𐎔", "𐎕", "𐎖", "𐎗", "𐎘", "𐎙", "𐎚", "𐎛", "𐎜", "𐎝"
]


def generate_char_image(char, font, margin=10):

    dummy_img = Image.new("L", (1, 1), 255)
    draw = ImageDraw.Draw(dummy_img)
    
    bbox = draw.textbbox((0, 0), char, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    width = text_width + 2 * margin
    height = text_height + 2 * margin
    img = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(img)
    
    draw.text((margin, margin), char, font=font, fill=0)
    return img

def compute_features(image, threshold=128):
    arr = np.array(image)
    h, w = arr.shape
    binary = (arr < threshold).astype(np.float32)

    if np.sum(binary) == 0:
        return np.zeros(5)

    mass = np.sum(binary)
    mass_norm = mass / (w * h)
    
    indices = np.argwhere(binary > 0)
    y_coords = indices[:, 0]
    x_coords = indices[:, 1]
    
    center_x = np.mean(x_coords) / w
    center_y = np.mean(y_coords) / h

    x_rel = x_coords / w - center_x
    y_rel = y_coords / h - center_y
    
    moment_x = np.mean(x_rel**2)
    moment_y = np.mean(y_rel**2)
    
    return np.array([mass_norm, center_x, center_y, moment_x, moment_y])

def closeness_measure(feat1, feat2):
    distance = np.linalg.norm(feat1 - feat2)
    return 1 / (1 + distance)

def build_templates(font):
    templates = {}
    for char in alphabet_chars:
        img = generate_char_image(char, font)
        features = compute_features(img)
        templates[char] = features
    return templates

def recognize_text(text_to_recognize, font):
    templates = build_templates(font)
    hypotheses = []
    best_chars = []
    
    for char in text_to_recognize:
        img = generate_char_image(char, font)
        feat = compute_features(img)
        curr_hypotheses = []
        
        for tmpl_char, tmpl_feat in templates.items():
            measure = closeness_measure(feat, tmpl_feat)
            curr_hypotheses.append((tmpl_char, measure))
        
        curr_hypotheses.sort(key=lambda x: x[1], reverse=True)
        hypotheses.append(curr_hypotheses)
        best_chars.append(curr_hypotheses[0][0])
    
    return hypotheses, "".join(best_chars)

def write_results_to_file(filename, hypotheses, true_text, best_text, error_count, percent_ok, experiment_results):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# Лабораторная работа №7: Классификация символов\n\n")
        f.write(f"## Исходная строка: `{true_text}`\n")
        f.write(f"## Распознанная строка: `{best_text}`\n")
        f.write(f"## Точность: {percent_ok:.2f}%\n\n")
        
        f.write("## Подробные гипотезы:\n")
        for i, hyp in enumerate(hypotheses, 1):
            f.write(f"{i}: {hyp[:5]}\n") 
        
        f.write("\n## Эксперимент с изменённым шрифтом\n")
        f.write(f"Размер шрифта: {experiment_results['font_size']}\n")
        f.write(f"Распознано: {experiment_results['best_text']}\n")
        f.write(f"Точность: {experiment_results['percent_ok']:.2f}%\n")

def main():
    font = ImageFont.truetype(font_path, default_font_size)
    true_text = "".join(alphabet_chars[:7]) 

    hypotheses, best_text = recognize_text(true_text, font)
    errors = sum(1 for t, b in zip(true_text, best_text) if t != b)
    accuracy = 100 * (len(true_text) - errors) / len(true_text)

    exp_font_size = default_font_size + 8
    exp_font = ImageFont.truetype(font_path, exp_font_size)
    exp_hypotheses, exp_best_text = recognize_text(true_text, exp_font)
    exp_errors = sum(1 for t, b in zip(true_text, exp_best_text) if t != b)
    exp_accuracy = 100 * (len(true_text) - exp_errors) / len(true_text)
    
    experiment_results = {
        "font_size": exp_font_size,
        "best_text": exp_best_text,
        "percent_ok": exp_accuracy
    }
    
    write_results_to_file(
        "lab_7_.md",
        hypotheses,
        true_text,
        best_text,
        errors,
        accuracy,
        experiment_results
    )
    
    print("Результаты сохранены в lab_7.md")

if __name__ == "__main__":
    main()