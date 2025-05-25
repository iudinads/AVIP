from PIL import Image
import numpy as np

def convolve(image_array, kernel):
    image_h, image_w = image_array.shape
    kernel_h, kernel_w = kernel.shape

    pad_h = kernel_h // 2
    pad_w = kernel_w // 2

    padded_image = np.pad(image_array, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

    output = np.zeros_like(image_array, dtype=np.float32)

    for i in range(image_h):
        for j in range(image_w):
            window = padded_image[i:i+kernel_h, j:j+kernel_w]
            output[i, j] = np.sum(window * kernel)
    return output

def normalize_array(arr):
    """
    norm = (arr - arr_min) / (arr_max - arr_min) * 255 (минимальное становится 0, максимальное - 255)
    """
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    norm = (arr - arr_min) / (arr_max - arr_min) * 255
    return norm.astype(np.uint8)

kernel_Gx = np.array([
    [-1, -1, -1, -1, -1],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 1,  1,  1,  1,  1]
], dtype=np.float32)

kernel_Gy = np.array([
    [-1,  0,  0,  0,  1],
    [-1,  0,  0,  0,  1],
    [-1,  0,  0,  0,  1],
    [-1,  0,  0,  0,  1],
    [-1,  0,  0,  0,  1]
], dtype=np.float32)

input_filename = "fox.png"

color_img = Image.open(input_filename)
color_img.save("fox1_img.png") 

gray_img = color_img.convert("L")  
gray_img.save("grayscale.png")
gray_array = np.array(gray_img, dtype=np.float32)


Gx = convolve(gray_array, kernel_Gx)
Gx_norm = normalize_array(Gx)  

Gy = convolve(gray_array, kernel_Gy)
Gy_norm = normalize_array(Gy)

# G = sqrt(Gx^2 + Gy^2)
G = np.sqrt(Gx**2 + Gy**2)
G_norm = normalize_array(G)

Image.fromarray(Gx_norm).save("Gx.png")
Image.fromarray(Gy_norm).save("Gy.png")
Image.fromarray(G_norm).save("G.png")

threshold = 25
G_binary = np.where(G_norm > threshold, 255, 0).astype(np.uint8)
Image.fromarray(G_binary).save("G_binary.png")

markdown_text = f"""# Лабораторная работа No4: Выделение контуров на изображении

**Вариант:** Оператор Приюитт 5x5

## 1. Исходное цветное изображение
![Исходное изображение](fox1_img.png)

## 2. Полутоновое изображение
![Полутоновое изображение](grayscale.png)

## 3. Градиентные матрицы

### 3.1. Горизонтальная градиентная матрица Gx (нормализована)
![Градиент Gx](Gx.png)

### 3.2. Вертикальная градиентная матрица Gy (нормализована)
![Градиент Gy](Gy.png)

### 3.3. Итоговая градиентная матрица G = sqrt(Gx^2 + Gy^2) (нормализована)
![Градиент G](G.png)

## 4. Бинаризованная градиентная матрица G
*Порог бинаризации: {threshold} (значения выше порога принимают значение 255, остальные 0)*
![Бинаризованное изображение G](G_binary.png)
"""

with open("lab_4.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)

print("Лабораторная работа выполнена")
