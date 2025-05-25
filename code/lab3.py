from PIL import Image
import numpy as np

def rank_filter_cross(image_array, rank=4):
    height, width = image_array.shape
    result = np.copy(image_array)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            cross = [
                image_array[i, j],      # центр
                image_array[i - 1, j],  # верх
                image_array[i + 1, j],  # низ
                image_array[i, j - 1],  # лево
                image_array[i, j + 1]   # право
            ]
            ones_count = sum(cross)
            
            # Решающее правило: Если количество единиц >= rank = 4 , то результирующий пиксель становится 1, иначе – 0.
            result[i, j] = 1 if ones_count >= rank else 0

    result = (result * 255).astype(np.uint8)

    return result

def compute_difference(original, filtered): 
    # если значения совпадают, то результат 0 черный , иначе 255 белый.
    diff = np.where(original == filtered, 0, 255)
    return diff.astype(np.uint8)


def main():
    png = ['image_3.png', 'image_2.png']
    count = 0
    for i in png:
        count += 1

        gray_image = Image.open(i).convert('L')
        
        img_array = np.array(gray_image)
        bin_array = np.where(img_array > 128, 1, 0)
        
        filtered_array = rank_filter_cross(bin_array, rank=4)

        difference_array = compute_difference(bin_array, filtered_array)

        filtered_image = Image.fromarray(filtered_array)
        a = 'filtered' + '_' + str(count) + '.png'
        filtered_image.save(a)
    
        difference_image = Image.fromarray(difference_array)
        b = 'difference' + '_' + str(count) + '.png'
        difference_image.save(b)
   
    with open("report_lab_3_.md", "w", encoding="utf-8") as f:
        f.write("# Лабораторная работа No3\n")
        f.write("## Фильтрация изображений и морфологические операции\n\n")
        f.write("**Метод:** Ранговый фильтр с маской прямой крест (окно 3x3, ранг 4)\n\n")
        f.write("### Этапы обработки:\n")

        f.write("1. Исходные изображения\n")
        f.write('![](image_3.png)\n\n')
        f.write('![](image_2.png)\n\n')

        f.write("2. Применение рангового фильтра к изображению.\n")
        f.write('![](filtered_1.png)\n\n')
        f.write('![](filtered_2.png)\n\n')
        
        f.write("3. Вычисление разностного изображения как абсолютной разности между исходным и отфильтрованным изображениями.\n")
        f.write('![](difference_1.png)\n\n')
        f.write('![](difference_2.png)\n\n')
    
    print("Лабораторная работа выполнена.")

if __name__ == '__main__':
    main()
