from PIL import Image, ImageOps
import numpy as np
import math
import matplotlib.pyplot as plt

def rgb_to_hsl(image):
    hsl_image = image.convert("HSV")
    return hsl_image

def hsl_to_rgb(hsl_image):
    rgb_image = hsl_image.convert("RGB")
    return rgb_image

def extract_l_channel(hsl_image):
    l_channel = hsl_image.split()[2]
    return l_channel

def combine_hsl(h_channel, s_channel, l_channel):
    return Image.merge("HSV", (h_channel, s_channel, l_channel))

def power_transform(l_channel, gamma=1.0):
    l_array = np.array(l_channel, dtype=np.float32) / 255.0
    l_transformed = np.power(l_array, gamma) * 255.0
    return Image.fromarray(l_transformed.astype(np.uint8))

def calculate_ngldm(image_array, d=1, levels=256):
    height, width = image_array.shape
    ngldm = np.zeros(levels, dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            current_val = image_array[y, x]
            neighbors = []
            
            # d = 2
            for dy in range(-d, d+1):
                for dx in range(-d, d+1):
                    if abs(dy) + abs(dx) <= d and (dy != 0 or dx != 0):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            neighbors.append(image_array[ny, nx])
            
            if neighbors:
                avg_diff = np.mean(np.abs(current_val - np.array(neighbors)))
                ngldm[current_val] += avg_diff
    
    return ngldm

def calculate_sne(ngldm):
    total = np.sum(ngldm)
    if total == 0:
        return 0
    k = np.arange(len(ngldm))
    sne = np.sum(ngldm[1:] / (k[1:]**2 + 1e-6)) / total  
    return sne

def calculate_lne(ngldm):
    total = np.sum(ngldm)
    if total == 0:
        return 0
    k = np.arange(len(ngldm))
    lne = np.sum(ngldm[1:] * k[1:]**2) / total 
    return lne

def plot_histogram(image_array, title, ax):
    ax.hist(image_array.flatten(), bins=256, range=(0, 255), color='gray', alpha=0.7)
    ax.set_title(title)
    ax.set_xlim(0, 255)

def main():
    image_path = "your_image.jpg"
    original_image = Image.open(image_path)

    hsl_image = rgb_to_hsl(original_image)
    h_channel, s_channel, l_channel = hsl_image.split()
    l_array = np.array(l_channel)
    
    gamma = 0.5 
    l_transformed = power_transform(l_channel, gamma)
    l_transformed_array = np.array(l_transformed)

    transformed_hsl = combine_hsl(h_channel, s_channel, l_transformed)
    transformed_image = hsl_to_rgb(transformed_hsl)

    d = 2  
    ngldm_original = calculate_ngldm(l_array, d)
    ngldm_transformed = calculate_ngldm(l_transformed_array, d)

    sne_original = calculate_sne(ngldm_original)
    lne_original = calculate_lne(ngldm_original)
    sne_transformed = calculate_sne(ngldm_transformed)
    lne_transformed = calculate_lne(ngldm_transformed)

    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(transformed_image)
    axes[0, 1].set_title(f"Transformed Image (γ={gamma})")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(l_channel, cmap='gray')
    axes[1, 0].set_title("Original L Channel")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(l_transformed, cmap='gray')
    axes[1, 1].set_title(f"Transformed L Channel (γ={gamma})")
    axes[1, 1].axis('off')

    plot_histogram(l_array, "Original L Channel Histogram", axes[2, 0])
    plot_histogram(l_transformed_array, f"Transformed L Channel Histogram (γ={gamma})", axes[2, 1])
    
    plt.tight_layout()

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(ngldm_original, 'k')
    ax1.set_title("Original NGLDM")
    ax1.set_xlabel("Gray level")
    ax1.set_ylabel("Average difference")
    
    ax2.plot(ngldm_transformed, 'k')
    ax2.set_title("Transformed NGLDM")
    ax2.set_xlabel("Gray level")
    ax2.set_ylabel("Average difference")
    
    plt.tight_layout()

    print("Texture features:")
    print(f"Original - SNE: {sne_original:.4f}, LNE: {lne_original:.4f}")
    print(f"Transformed - SNE: {sne_transformed:.4f}, LNE: {lne_transformed:.4f}")

    fig.savefig("lab8_images.png")
    fig2.savefig("lab8_ngldm.png")

    with open("lab_8.md", "w") as f:
        f.write("# Лабораторная работа №8. Текстурный анализ и контрастирование\n\n")
        f.write("## Результаты\n\n")
        f.write("### Исходные и преобразованные изображения\n")
        f.write("![Изображения](lab8_images.png)\n\n")
        f.write("### Матрицы NGLDM\n")
        f.write("![NGLDM](lab8_ngldm.png)\n\n")
        f.write("### Текстурные признаки\n")
        f.write(f"- Исходное изображение: SNE = {sne_original:.4f}, LNE = {lne_original:.4f}\n")
        f.write(f"- Преобразованное изображение: SNE = {sne_transformed:.4f}, LNE = {lne_transformed:.4f}\n\n")
        f.write("### Выводы\n")
        f.write("В ходе работы было проведено степенное преобразование яркости изображения. ")
        f.write("Анализ текстурных признаков показывает, как изменились характеристики текстуры после преобразования.\n")
    
    plt.show()

if __name__ == "__main__":
    main()