from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import csv
import os


font_path = "Noto_Sans_Ugaritic/NotoSansUgaritic-Regular.ttf" 
font_size = 52
chars = [
    "𐎀", "𐎁", "𐎂", "𐎃", "𐎄", "𐎅", "𐎆", "𐎇", "𐎈", "𐎉",
    "𐎊", "𐎋", "𐎌", "𐎍", "𐎎", "𐎏", "𐎐", "𐎑", "𐎒", "𐎓",
    "𐎔", "𐎕", "𐎖", "𐎗", "𐎘", "𐎙", "𐎚", "𐎛", "𐎜", "𐎝"
]
output_folder = "symbols"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

font = ImageFont.truetype(font_path, font_size)
for i, char in enumerate(chars):
    image = Image.new("L", (font_size * 2, font_size * 2), 255) 
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), char, font=font, fill=0) 
    image = image.crop(image.getbbox())
    image.save(f"{output_folder}/symbol_{i}.png") 

def calculate_features(image):
    data = np.array(image)
    black_pixels = np.where(data == 0)
    n = len(black_pixels[0])

    h, w = data.shape
    quarters = [
        data[:h//2, :w//2], data[:h//2, w//2:],
        data[h//2:, :w//2], data[h//2:, w//2:]
    ]
    weights = [np.sum(q == 0) for q in quarters]

    area = (h * w) / 4
    specific_weights = [w / area for w in weights]

    x_c = np.mean(black_pixels[1])
    y_c = np.mean(black_pixels[0])

    x_norm = x_c / w
    y_norm = y_c / h

    I_x = np.sum((black_pixels[0] - y_c) ** 2)
    I_y = np.sum((black_pixels[1] - x_c) ** 2)

    I_x_norm = I_x / (h * w)
    I_y_norm = I_y / (h * w)

    x_profile = np.sum(data == 0, axis=0)
    y_profile = np.sum(data == 0, axis=1)

    return {
        "weights": weights,
        "specific_weights": specific_weights,
        "x_c": x_c, "y_c": y_c,
        "x_norm": x_norm, "y_norm": y_norm,
        "I_x": I_x, "I_y": I_y,
        "I_x_norm": I_x_norm, "I_y_norm": I_y_norm,
        "x_profile": x_profile, "y_profile": y_profile
    }

with open("features.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile, delimiter=";")
    writer.writerow(["Символ", "Вес_1", "Вес_2", "Вес_3", "Вес_4",
                     "Удельный_вес_1", "Удельный_вес_2", "Удельный_вес_3", "Удельный_вес_4",
                     "X_c", "Y_c", "X_norm", "Y_norm", "I_x", "I_y", "I_x_norm", "I_y_norm"])

    with open("report_5.md", "w", encoding="utf-8") as report_file:
        report_file.write("# Лабораторная работа №5. Выделение признаков символов\n\n")

        for i, char in enumerate(chars):
            image = Image.open(f"{output_folder}/symbol_{i}.png").convert("L")
            features = calculate_features(image)

            writer.writerow([char, *features["weights"], *features["specific_weights"],
                             features["x_c"], features["y_c"], features["x_norm"], features["y_norm"],
                             features["I_x"], features["I_y"], features["I_x_norm"], features["I_y_norm"]])

            plt.bar(range(len(features["x_profile"])), features["x_profile"])
            plt.title(f"X Profile for {char}")
            plt.savefig(f"{output_folder}/symbol_{i}_x_profile.png")
            plt.clf()

            plt.bar(range(len(features["y_profile"])), features["y_profile"])
            plt.title(f"Y Profile for {char}")
            plt.savefig(f"{output_folder}/symbol_{i}_y_profile.png")
            plt.clf()

            report_file.write(f"## Символ {char}\n\n")
            report_file.write(f"![Изображение]({output_folder}/symbol_{i}.png)\n\n")
            report_file.write(f"### Профили:\n")
            report_file.write(f"![Профиль X]({output_folder}/symbol_{i}_x_profile.png)\n")
            report_file.write(f"![Профиль Y]({output_folder}/symbol_{i}_y_profile.png)\n\n")
            report_file.write(f"### Признаки:\n")
            report_file.write(f"- Вес каждой четверти: {features['weights']}\n")
            report_file.write(f"- Удельный вес: {features['specific_weights']}\n")
            report_file.write(f"- Центр тяжести: ({features['x_c']:.2f}, {features['y_c']:.2f})\n")
            report_file.write(f"- Нормированные координаты центра тяжести: ({features['x_norm']:.4f}, {features['y_norm']:.4f})\n")
            report_file.write(f"- Моменты инерции: (X: {features['I_x']:.2f}, Y: {features['I_y']:.2f})\n")
            report_file.write(f"- Нормированные моменты инерции: (X: {features['I_x_norm']:.4f}, Y: {features['I_y_norm']:.4f})\n\n")

print("Обработка завершена. Результаты сохранены в файлы features.csv и report_5.md.")