from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import os


font_path = "Noto_Sans_Ugaritic/NotoSansUgaritic-Regular.ttf"
font_size = 52
phrase = "𐎀𐎁𐎂 𐎃𐎄𐎅" 
chars = ["𐎀", "𐎁", "𐎂", "𐎃", "𐎄", "𐎅", "𐎆", "𐎇", "𐎈", "𐎉"]


font = ImageFont.truetype(font_path, font_size)


png_image_path = "text.png" 
bmp_image_path = "phrase.bmp"  


image = Image.open(png_image_path).convert("1")  
image.save(bmp_image_path, format='BMP')


def calculate_profiles(image):
    data = np.array(image)
    vertical_profile = np.sum(data == 0, axis=0)  
    horizontal_profile = np.sum(data == 0, axis=1) 
    return vertical_profile, horizontal_profile


image = Image.open("phrase.bmp").convert("L") 
v_profile, h_profile = calculate_profiles(image)


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(len(v_profile)), v_profile)
plt.title("Вертикальный профиль")
plt.subplot(1, 2, 2)
plt.barh(range(len(h_profile)), h_profile)
plt.title("Горизонтальный профиль")
plt.savefig("profiles.png")
plt.close()


def segment_characters(image, threshold=2):
    data = np.array(image)
    v_profile = np.sum(data == 0, axis=0) 
    
    in_char = False
    start_pos = []
    end_pos = []
    
    for i, val in enumerate(v_profile):
        if val > threshold and not in_char:
            start_pos.append(i)
            in_char = True
        elif val <= threshold and in_char:
            end_pos.append(i)
            in_char = False
    
    if in_char:
        end_pos.append(len(v_profile)-1)
    
    bboxes = []
    for start, end in zip(start_pos, end_pos):
        char_slice = data[:, start:end]
        h_profile = np.sum(char_slice == 0, axis=1)
        top = np.argmax(h_profile > 0) 
        bottom = len(h_profile) - np.argmax(h_profile[::-1] > 0) - 1 
        bboxes.append((start, top, end, bottom))
    
    return bboxes


bboxes = segment_characters(image)


result = image.copy().convert("RGB")
draw = ImageDraw.Draw(result)
for bbox in bboxes:
    draw.rectangle(bbox, outline="red")
result.save("segmented.png")

if not os.path.exists("symbol_profiles"):
    os.makedirs("symbol_profiles")

for i, char in enumerate(chars):
    char_image = Image.new("L", (font_size*2, font_size*2), 255)
    draw = ImageDraw.Draw(char_image)
    draw.text((10, 10), char, font=font, fill=0)
    char_image = char_image.crop(char_image.getbbox())  
    
    v_profile, h_profile = calculate_profiles(char_image)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(v_profile)), v_profile)
    plt.title(f"Vertical profile {i}") 
    plt.subplot(1, 2, 2)
    plt.barh(range(len(h_profile)), h_profile)
    plt.title(f"Horizontal profile {i}")
    plt.tight_layout()
    plt.savefig(f"symbol_profiles/profile_{i}.png")
    plt.close()

with open("lab_6_.md", "w", encoding="utf-8") as report:
    report.write("# Лабораторная работа №6: Сегментация текста\n\n")
    
    report.write("## 1. Исходное изображение фразы\n")
    report.write("![Фраза](phrase.bmp)\n\n")
    
    report.write("## 2. Профили изображения\n")
    report.write("![Профили](profiles.png)\n\n")
    
    report.write("## 3. Результат сегментации\n")
    report.write("![Сегментация](segmented.png)\n\n")
    report.write("Координаты символов:\n")
    for i, bbox in enumerate(bboxes):
        report.write(f"- Символ {i+1}: {bbox}\n")
    report.write("\n")
    
    report.write("## 4. Профили символов алфавита\n")
    for i, char in enumerate(chars):
        report.write(f"### Символ {char} (индекс {i})\n")
        report.write(f"![Профили символа {i}](symbol_profiles/profile_{i}.png)\n\n")