# Загрузка библиотек

import os
import math                               # Математические функции
import random                             # Генерация случайных чисел
import numpy as np                        # Работа с массивами
import matplotlib.pyplot as plt           # Отрисовка графиков
# Инструменты для работы с изображениями
from PIL import Image, ImageEnhance
# # Инструменты для работы с изображениями
from tensorflow.keras.preprocessing import image


# Служебная функция загрузки выборки изображений из файлов в папке

def load_imageset(folder,   # имя папки
                  subset,   # подмножество изображений - оригинальные или сегментированные
                  title     # имя выборки
                  ):

    # Cписок для хранения изображений выборки
    image_list = []

    # Для всех файлов в каталоге по указанному пути:
    for filename in sorted(os.listdir(f'{folder}/{subset}')):

        # Чтение очередной картинки и добавление ее в список изображений требуемого размера
        image_list.append(image.load_img(os.path.join(f'{folder}/{subset}', filename),
                                         target_size=(IMG_WIDTH, IMG_HEIGHT)))

    # Вывод количества элементов в выборке
    print('Количество изображений:', len(image_list))

    return image_list


# Глобальные параметры
IMG_WIDTH = 512               # Ширина картинки
IMG_HEIGHT = 512              # Высота картинки

TRAIN_DIRECTORY = './dataset/train'  # Путь до папки с файлами обучающей выборки
# Загрузка входных изображений
train_images = load_imageset(TRAIN_DIRECTORY, 'origa', 'Обучающая')
# Загрузка выходных (сегментированных) изображений
train_segments = load_imageset(TRAIN_DIRECTORY, 'segment', 'Обучающая')


# Функции аугментации

def show_image_pair(img1, img2):
    # Cоздание полотна для рисования двух изображений
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Исходное изображение
    axs[0].imshow(img1)
    axs[0].axis('off')

    # Модифицированное изображение
    axs[1].imshow(img2)
    axs[1].axis('off')

    # Вывод изображений
    plt.show()


def augment_image(img,                    # Изображение для аугментации
                  segm,                   # Маска
                  ang=15,                 # Максимальный угол поворота
                  f_x=0.08,               # Максимальная подрезка по ширине
                  f_y=0.08,               # Максимальная подрезка по высоте
                  level_contr=0.3,        # Максимальное отклонение коэффициента контраста от нормы
                  level_brght=0.3):       # Максимальное отклонение коэффициента яркости от нормы

    # Функция нахождения ширины и высоты прямоугольника наибольшей площади
    # после поворота заданного прямоугольника на угол в градусах

    def rotated_rect(w, h, angle):
        angle = math.radians(angle)
        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
            x = 0.5 * side_short
            wr, hr = (x/sin_a, x/cos_a) if width_is_longer else (x/cos_a, x/sin_a)
        else:
            cos_2a = cos_a*cos_a - sin_a*sin_a
            wr, hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

        return wr, hr

    # Функция случайной обрезки

    def random_crop(x,                    # Подаваемое изображение
                    y,
                    # Предел обрезки справа и слева (в масштабе ширины)
                    f_x=f_x,
                    # Предел обрезки сверху и снизу (в масштабе высоты)
                    f_y=f_x
                    ):

        # Получение левой и правой границ обрезки
        left = x.width * random.random() * f_x
        right = x.width * (1. - random.random() * f_x) - 1.

        # Получение верхней и нижней границ обрезки
        upper = x.height * random.random() * f_y
        lower = x.height * (1. - random.random() * f_y) - 1.

        return x.crop((left, upper, right, lower)), y.crop((left, upper, right, lower)),

    # Функция случайного поворота

    def random_rot(x,                     # Исходное
                   y,                     # Маска
                   ang=ang                # Максимальный угол поворота
                   ):

        # Случайное значение угла в диапазоне [-ang, ang]
        a = random.uniform(-1., 1.) * ang

        # Вращение картинки с расширением рамки
        r1 = x.rotate(a, expand=True)
        r2 = y.rotate(a, expand=True)

        # Вычисление размеров прямоугольника обрезки максимальной площади
        # для размеров исходной картинки и угла поворота в градусах
        crop_w1, crop_h1 = rotated_rect(x.width, x.height, a)
        crop_w2, crop_h2 = rotated_rect(y.width, y.height, a)

        # Обрезка повернутого изображения и возврат результата
        w1, h1 = r1.size
        xxx = r1.crop(((w1 - crop_w1)*0.5, (h1 - crop_h1)*0.5,
                      (w1 + crop_w1)*0.5, (h1 + crop_h1)*0.5))
        w2, h2 = r2.size
        yyy = r2.crop(((w2 - crop_w2)*0.5, (h2 - crop_h2)*0.5,
                      (w2 + crop_w2)*0.5, (h2 + crop_h2)*0.5))

        return xxx, yyy

    # Функция случайного изменения контрастности

    def random_contrast(x,                   # Подаваемое изображение
                        # Максимальное отклонение коэффициента контраста от нормы - число от 0. до 1.
                        level=level_contr
                        ):

        # Создание экземпляра класса Contrast
        enh = ImageEnhance.Contrast(x)
        factor = random.uniform(1. - level,
                                1. + level)  # Cлучайный коэффициент контраста из указанного интервала

        return enh.enhance(factor)           # Изменение коэффициента контраста

    # Функция случайного изменения яркости

    def random_brightness(x,                 # Подаваемое изображение
                          # Максимальное отклонение коэффициента яркости от нормы - число от 0. до 1.
                          level=level_brght
                          ):

        # Создание экземпляра класса Brightness
        enh = ImageEnhance.Brightness(x)
        factor = random.uniform(1. - level,
                                1. + level)  # Cлучайный коэффициент контраста из указанного интервала

        return enh.enhance(factor)           # Изменение коэффициента яркости

    # Функция отражения

    def trans_img(x, y):
        return x.transpose(Image.FLIP_LEFT_RIGHT), y.transpose(Image.FLIP_LEFT_RIGHT)

    # Тело основной функции

    # Cоздание списка модификаций для изображения
    mod_oper = [random_rot,
                random_crop,
                trans_img,
                random_contrast,
                random_brightness]

    # Количество изменений из списка; применяем все
    mod_count = 5

    # Случайный отбор индексов изменений в количестве mod_count без повторений
    mod_list = random.sample(range(len(mod_oper)), mod_count)

    # Применение модификаций по индексам из mod_list
    for mod_index in mod_list:
        if mod_index < 3:
            img, segm = mod_oper[mod_index](img, segm)
        else:
            img = mod_oper[mod_index](img)

    # Возврат результата
    return img, segm


# Аугментация
k = 1000  # Задаем начальное значение, чтобы при сохарнении не пересечься с неаугментированными изображениями
for i in range(0, len(train_images)):
    augment_origa, augment_segment = augment_image(
        train_images[i], train_segments[i])
    augment_origa.save(
        os.path.join(TRAIN_DIRECTORY, f'origa/{k}.png'), quality=95)
    augment_segment.save(
        os.path.join(TRAIN_DIRECTORY, f'segment/{k}.png'), quality=95)
    k += 1
