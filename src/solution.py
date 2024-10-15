from pathlib import Path
import random
import numpy as np
from typing import Union, List
import torch
from ultralytics import YOLO


# Фиксируем сиды
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).absolute().parent.parent

if torch.cuda.is_available():
    torch.device('cuda')
    DEVICE = 'cuda'
else:
    torch.device('cpu')
    DEVICE = 'cpu'

# Загрузка модели YOLOv8
MODEL = YOLO(Path(BASE_DIR, 'model', 'v7_best.pt')).to(DEVICE)


def predict(images: Union[str, Path, List[np.ndarray], np.ndarray]) -> list[dict]:
    """

    :param images:
    :return:
        list[dict]: список списков словарей с результатами предикта
        на найденных изображениях [
            [
                {
                    'xc': round(xc, 4),
                    'yc': round(yc, 4),
                    'w': round(w, 4),
                    'h': round(h, 4),
                    'label': 0,
                    'score': round(conf, 4),
                },
                ...
            ],
                ...
            [
                {
                    'xc': round(xc, 4),
                    'yc': round(yc, 4),
                    'w': round(w, 4),
                    'h': round(h, 4),
                    'label': 0,
                    'score': round(conf, 4),
                },
                ...
            ]
        ]
    """
    results = []
    model_inference = MODEL(images)  # return a list of Results objects

    # Process results list
    num_images = len(images)
    for image_num in range(num_images):
        temp_results = []
        answers = model_inference[image_num].boxes
        num_found = answers.shape[0]
        for i in range(num_found):
            xc, yc, w, h = answers[i].xywh.numpy().reshape(-1)
            conf = answers[i].conf.numpy()[0]
            cls = answers[i].cls.numpy()[0]
            result = {
                "xc": round(xc, 4),
                "yc": round(yc, 4),
                "w": round(w, 4),
                "h": round(h, 4),
                "label": cls,
                "score": round(conf, 4),
            }
            temp_results.append(result)
        if len(temp_results) == 0:
            result = {
                "xc": None,
                "yc": None,
                "w": None,
                "h": None,
                "label": None,
                "score": None,
            }
            temp_results.append(result)
        results.append(temp_results)

    return results


if __name__ == "__main__":
    import cv2

    img = cv2.imread(r'../test_images/1_002301_zoom.JPG')
    img1 = cv2.imread(r'../test_images/1_000178.JPG')

    predict([img, img1])
