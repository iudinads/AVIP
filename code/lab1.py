from PIL import Image
import numpy as np
import math

input_filename = 'image.png' 
image = Image.open(input_filename) 
image = image.convert('RGB')
#arr = np.array(image)

def image_to_np_array(image_name: str) -> np.array:
    return np.array(image_name)


def save_image(image_array, filename):
    img = Image.fromarray(np.uint8(image_array))  # Преобразование в формат изображения
    img.save(filename)


def extract_rgb_components(image_name):
    img = image_to_np_array(image_name)

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    save_image(np.stack([R, np.zeros_like(R), np.zeros_like(R)], axis=2), 'Output_R.png')
    save_image(np.stack([np.zeros_like(G), G, np.zeros_like(G)], axis=2), 'Output_G.png')
    save_image(np.stack([np.zeros_like(B), np.zeros_like(B), B], axis=2), 'Output_B.png')

extract_rgb_components(image)

# Сохраняем полученные изображения
#red_img.save('Output_R.png')
#green_img.save('Output_G.png')
#blue_img.save('Output_B.png')

def rgb_to_hsi(image):
    img_array = np.array(image).astype(np.float32) / 255.0 
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]

    I = (R + G + B) / 3.0

    min_rgb = np.minimum(np.minimum(R, G), B)
    S = 1 - (min_rgb / (I + 1e-10))
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-10
    theta = np.arccos(np.clip(num / den, -1, 1))

    H = np.zeros_like(I)
    H[B > G] = (2 * np.pi - theta[B > G])
    H[B <= G] = theta[B <= G]
    H = H / (2 * np.pi)

    return H, S, I

H, S, I = rgb_to_hsi(image)

I_image = Image.fromarray((I * 255).astype(np.uint8))
I_image.save('Output_Intensity.png')

I_inverted = 1 - I
I_inverted_img = Image.fromarray((I_inverted * 255).astype(np.uint8))
I_inverted_img.save('Output_Inverted_Intensity.png')


def stretch_image(image, M):
    src_width, src_height = image.size
    new_width = src_width * M
    new_height = src_height * M
    new_image = Image.new(image.mode, (new_width, new_height)) #image.mode - RGB

    for new_y in range(new_height):
        for new_x in range(new_width):
            orig_x = new_x // M
            orig_y = new_y // M
            new_image.putpixel((new_x, new_y), image.getpixel((orig_x, orig_y)))
    return new_image

M = 3
stretched_img = stretch_image(image, M)
stretched_img.save('Output_Stretched.png')



def compress_image(image, N):
    src_width, src_height = image.size
    new_width = src_width // N
    new_height = src_height // N
    new_image = Image.new(image.mode, (new_width, new_height))

    for new_y in range(new_height):
        for new_x in range(new_width):
            orig_x = new_x * N
            orig_y = new_y * N
            new_image.putpixel((new_x, new_y), image.getpixel((orig_x, orig_y)))
    return new_image

N = 2 
compressed_img = compress_image(image, N)
compressed_img.save('Output_Compressed.png')

def two_pass_resample(image, M, N):
    M = 3
    N = 2
    stretched = stretch_image(image, M)
    result = compress_image(stretched, N)
    return result

K = M / N  
two_pass_img = two_pass_resample(image, M, N)
two_pass_img.save('Output_TwoPass_Resampled.png')

def one_pass_resample(image, K):
    src_width, src_height = image.size

    new_width = int(src_width / K)
    new_height = int(src_height / K)
    new_image = Image.new(image.mode, (new_width, new_height))

    for new_y in range(new_height):
        for new_x in range(new_width):
            orig_x = int(new_x * K)
            orig_y = int(new_y * K)
            new_image.putpixel((new_x, new_y), image.getpixel((orig_x, orig_y)))
    return new_image

one_pass_img = one_pass_resample(image, K)
one_pass_img.save('Output_OnePass_Resampled.png')

report_text = f"""# Отчет по лабораторной работе

## 1. Выделение компонент R, G, B
Были разделены каналы изображения. Ниже представлены результаты:
- Канал R: ![](Output_R.png)
- Канал G: ![](Output_G.png)
- Канал B: ![](Output_B.png)

## 2. Преобразование изображения в HSI
Ниже представлены компоненты:
- Яркостная компонента: ![](Output_Intensity.png)
- Инвертированная яркостная компонента: ![](Output_Inverted_Intensity.png)

## 3. Изменение размера изображения

### 3.1 Изначальные размеры изображения
Размер изображения: {image.size}

### 3.2 Растяжение (интерполяция)
Коэффициент растяжения: M = {M}  
Результат растяжения: ![](Output_Stretched.png)  
Размер изображения: {stretched_img.size}

### 3.3 Сжатие (децимация)
Коэффициент сжатия: N = {N}  
Результат сжатия: ![](Output_Compressed.png)  
Размер изображения: {compressed_img.size}

### 3.4 Передискретизация в два прохода (растяжение + сжатие)
Коэффициент: K = M / N = {K}  
Результат передискретизации в два прохода: ![](Output_TwoPass_Resampled.png)  
Размер изображения: {two_pass_img.size}

### 3.5 Передискретизация в один проход
Коэффициент: K = {K}  
Результат передискретизации в один проход: ![](Output_OnePass_Resampled.png)  
Размер изображения: {one_pass_img.size}
"""

with open('report_lab1_исп__.md', 'w', encoding='utf-8') as report_file:
    report_file.write(report_text)


if __name__ == "__main__":
    print("Лабораторная работа выполнена Полученные файлы с изображениями:")
    print("Output_R.png, Output_G.png, Output_B.png, Output_Intensity.png, Output_Inverted_Intensity.png")
    print("Output_Stretched.png, Output_Compressed.png, Output_TwoPass_Resampled.png, Output_OnePass_Resampled.png")
    print("Отчет сохранен в файле report_lab1.md")
    print(f'M = {M}, N = {N}, K = {K}')
