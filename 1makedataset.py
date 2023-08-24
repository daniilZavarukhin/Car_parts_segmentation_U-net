
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import requests
from pycocotools.coco import COCO
from skimage import io, data
import skimage.io as io
import PIL
from PIL import Image, ImageDraw


# создание директорий под будущий dataset
dir_list = ["./dataset",
            "./dataset/train", "./dataset/train/origa", "./dataset/train/segment",
            "./dataset/val", "./dataset/val/origa", "./dataset/val/segment"]
for dir in dir_list:
    if not os.path.isdir(dir):
        os.mkdir(dir)


# задаем цвета категориям, для дальнейшего использования
cats_short = {10: "#000000", 20: "#fbe901", 30: "#4a1a80", 40: "#005dad", 50: "#e6092a",
              60: "#005dad", 70: "#e6092a", 80: "#1aa737", 90: "#ed5911", 100: "#96c320",
              110: "#95117f",  120: "#96c320", 130: "#95117f", 140: "#ffaec8", 150: "#099ea2",
              160: "#099ea2", 170: "#c4ff0e", 180: "#f5ae08", 190: "#f0f0f0"}

# расширяем количество ключей по цветам, чтобы использовать неоднократное использовние ключа
cats = {}
for ss, mass in cats_short.items():
    for i in range(0, 6):
        cats[ss+i] = mass
# print(cats)


def save_pict(start_dir, finish_dir):
    # подгружаем аннотации для дальнейшего использования
    annotation_file = '{}/annotations.json'.format(start_dir)
    coco = COCO(annotation_file)
    imgIds = coco.getImgIds()
    images = coco.loadImgs(imgIds)
    # print(imgIds)

    # сохраняем оригиналы
    for k in imgIds:
        img = coco.loadImgs(k)[0]
        I = io.imread('{}/{}'.format(start_dir, img['path']))   # /255.0
        io.imsave(os.path.join(finish_dir, f'origa/{k}.png'), I)

    # сохраняем сегментированные
    for k in imgIds:
        img = coco.loadImgs(k)[0]
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)

        # объединяем классовую сегментацию, чтобы в дальнейшем избежать перезаписи
        iii = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19]
        category_id = []
        for i in anns:
            category_id.append(i['category_id'])
    #  print(category_id)

        for ii in iii:
            x = [i for i, ltr in enumerate(category_id) if ltr == ii]
            if len(x) > 1:
                anns[x[0]]['segmentation'].append(
                    anns[x[1]]['segmentation'][0])
                anns.pop(x[1])
            # print(anns[x[0]]['segmentation'])

        category_id = []
        for i in anns:
            category_id.append(i['category_id'])
        # print(category_id)

        # создаем словарь на основе аннотации, где ключами является категория, а его значение это точки для построения сегментации.
        # также сортируем ключи, для того чтобы мелкие объекты не замазались более крупными при сегментаии
        reanns = {}
        kkk = []
        for i in anns:
            if len(i['segmentation']) != 1:
                reanns[10*int(i['category_id'])] = i['segmentation'][0]
                reanns[10*int(i['category_id'])+1] = i['segmentation'][1]
            else:
                reanns[10*int(i['category_id'])] = i['segmentation'][0]

        reanns1 = {}
        ssss = sorted(list(reanns))

        for i in ssss:
            reanns1[i] = reanns[i]

        # преобразуем списки координат в пары координат x и y, для дальнейшего использования
        reanns2 = {}
        for ss, mass in reanns1.items():
            mmmm = []
            for i in range(0, len(mass)-1, 2):
                m = []
                m.append(mass[i])
                m.append(mass[i+1])
                mmmm.append(tuple(m))
            reanns2[ss] = tuple(mmmm)

        # наносим сегментацию на холст, где массивы используются как координаты, а их ключи как к списку цветов
        im = Image.new('RGB', (img['width'], img['height']), (0, 0, 0))
        draw = ImageDraw.Draw(im)

        for ss, mass in reanns2.items():
            draw.polygon(xy=mass, fill=cats[ss])
        im.save(os.path.join(finish_dir, f'segment/{k}.png'), quality=95)


# создаем проверочную выборку
save_pict('./Car-Parts-Segmentation-master/testset', './dataset/val')

# создаем тренировочную выборку
save_pict('./Car-Parts-Segmentation-master/trainingset', './dataset/train')
