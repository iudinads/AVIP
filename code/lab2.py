import numpy as np
from PIL import Image

k = 0.2
R = 128.0
window_size = 5 

def convert_to_grayscale(color_image: np.ndarray) -> np.ndarray:
    grayscale = np.dot(color_image[..., :3], [0.3, 0.59, 0.11])
    return grayscale.astype(np.uint8)

def compute_local_statistics(image: np.ndarray, i: int, j: int, window_size: int) -> (float, float, float):
    H, W = image.shape
    half_win = window_size // 2
    window = []
    
    for di in range(-half_win, half_win + 1):
        for dj in range(-half_win, half_win + 1):
            ni = min(max(i + di, 0), H - 1)
            nj = min(max(j + dj, 0), W - 1)
            window.append(image[ni, nj])
            
    window = np.array(window, dtype=np.float32)
    mean = np.mean(window)
    std = np.std(window)
    max_val = np.max(window)
    
    return mean, std, max_val

def adaptive_binarization(gray_array, window_size, k, R):
    H, W = gray_array.shape

    binary_arr = np.zeros((H, W), dtype=np.uint8)
    
    for i in range(H):
        for j in range(W):
            m, s, local_max = compute_local_statistics(gray_array, i, j, window_size)
            m_max = (local_max + m) / 2.0
            T = m_max * (1 - k * (1 - s / R))
            
            if gray_array[i, j] >= T:
                binary_arr[i, j] = 255 
            else:
                binary_arr[i, j] = 0 
    
    return binary_arr

def main():
    count = 0
    for i in ['book.png', 'fingers.png', 'cat_2.png']:
        count += 1

        color_img = Image.open(i).convert('RGB')
        color_array = np.array(color_img)
    
        grayscale = convert_to_grayscale(color_array)
        grayscale_img = Image.fromarray(grayscale)
        a = 'grayscale' + '_' + str(count) + '.bmp'
        grayscale_img.save(a)
    
        binary = adaptive_binarization(grayscale, window_size, k, R) 
        binary_img = Image.fromarray(binary)
        b = 'binary' + '_' + str(count) + '.bmp'
        binary_img.save(b)

    with open('2_lab_.md', 'w', encoding='utf-8') as f:
        f.write('# Лабораторная работа Nº2: Обесцвечивание и бинаризация растровых изображений\n\n')

        f.write('## Исходные изображения: \n\n')

        f.write('![](book.png)\n\n')
        f.write('![](fingers.png)\n\n')
        f.write('![](cat_2.png)\n\n')

        f.write('## Этап 1: Приведение полноцветного изображения к полутоновому\n')

        f.write('![Полутоновое изображение](grayscale_1.bmp)\n\n')
        f.write('![Полутоновое изображение](grayscale_2.bmp)\n\n')
        f.write('![Полутоновое изображение](grayscale_3.bmp)\n\n')
        f.write('  \n')
        f.write('Формула вычисления яркости каждого пикселя:\n')
        f.write('  Y = 0.3 · R + 0.59 · G + 0.11 · B\n\n')
            
        f.write('## Этап 2: Приведение полутонового изображения к монохромному методом пороговой обработки\n')
        f.write('- **Метод:** Адаптивная бинаризация WAN\n')
        f.write('- **Размер окна:** 5×5\n')
 
        f.write('- Порог вычисляется по формуле:\n')
        f.write('T = m_max * (1 - k * (1 - s/R))\n')
        f.write('m_max = (max + m) / 2\n')

        f.write('![Бинарное изображение](binary_1.bmp)\n')
        f.write('![Бинарное изображение](binary_2.bmp)\n')
        f.write('![Бинарное изображение](binary_3.bmp)\n')
        
        
    print('Лабораторная работа выполнена.')

if __name__ == '__main__':
    main()