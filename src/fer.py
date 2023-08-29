from typing import Any
import cv2
import torch
import numpy as np
from nn import NN 


CLASSIFIER_PATH = "./models/haarcascade_frontalface_default.xml"
NN_W_PATH = "./models/FER_trained_model.pt"

EMOTION_DICT = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disguest', 6: 'fear'}

DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class FaceDetectionGenerator:
    def __init__(self, id: int, v_path: str, s_path: str, c_path: str, max_num_frame: int): # s_path: str, c_path: str, fps: int, frame_size: tuple | list, max_num_frame: int):
        self.id = id
        self.vc = cv2.VideoCapture(v_path)
        fw = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        fh = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        vc_fps = self.vc.get(cv2.CAP_PROP_FPS)
        print(f"VC {id} - FPS = {vc_fps}")
        # fourcc = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # self.vw = cv2.VideoWriter(s_path, 0, 10.0, (fw, fh))
        self.vw = cv2.VideoWriter(s_path, fourcc, vc_fps * 2.0, (fw, fh))
        self.face_cascade = cv2.CascadeClassifier(c_path)
        self.max_num_frame = max_num_frame
    
    async def __call__(self):
        if self.vc.isOpened():
            rval, frame = self.vc.read()
        else:
            rval = False
        
        while rval:
            # номер кадра
            frame_num = self.vc.get(cv2.CAP_PROP_POS_FRAMES)
            # позиция кадра во времени в миллисекундах
            frame_time = self.vc.get(cv2.CAP_PROP_POS_MSEC)
            # перевод цветов в оттенки серого
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # обнаружение лиц
            faces_loc = self.face_cascade.detectMultiScale(frame)

            face_lst = []
            for (x, y, w, h) in faces_loc:
                # добавление прямоугольника рамки в картинку
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                # извлечение изображения лица по рамке,
                # уменьшение размера картинки до 48*48 и нормализация
                face_lst.append(cv2.resize(gray[y:y + h, x:x + w], (48, 48))[None,None,:,:] / 256.0)

            yield {
                "id":self.id, 
                "frame num":frame_num,
                "frame time":frame_time,
                "marked frame":frame, 
                "detected faces":face_lst, 
                "faces loc":faces_loc,
                "pred values":[],
                "pred classes":[]
            }
            
            # if self.vw.isOpened():
            self.vw.write(frame)
            rval, frame = self.vc.read()
                
            if frame_num == self.max_num_frame:
                break
        
        self.vc.release()
        self.vw.release()


class FERNN_model:
    def __init__(self, model_path, em_dict, device=None, dtype=None):
        self.model = NN(device=device, dtype=dtype)
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
        self.emotion_dict = em_dict

    def __call__(self, img):
        pred_cls = []
        with torch.no_grad():
            self.model.eval()
            log_ps = self.model(img).cpu()
            ps = torch.exp(log_ps)
            top_v, top_class = ps.topk(1, dim=1)
            for c in top_class.numpy().ravel():
                pred_cls.append(self.emotion_dict[c])
            return top_v.numpy().ravel(), pred_cls

async def main(ulf_path_list: list, pf_path_list: list) -> None:
    net = FERNN_model(NN_W_PATH, EMOTION_DICT, DEVICE, DTYPE)

    fd_lst = [
        FaceDetectionGenerator(i, p, pf_path_list[i], CLASSIFIER_PATH, 50)()
        for i, p in enumerate(ulf_path_list)
    ]
    
    for _ in range(50):
        id_lst = []
        face_lst = []
        faces_locs = []
        marked_frame_lst = []

        for g in fd_lst:
            try:
                result = await anext(g)
                for idx, df in enumerate(result["detected faces"]):
                    id_lst.append(result["id"])
                    face_lst.append(df)
                    faces_locs.append(result["faces loc"][idx])
                marked_frame_lst.append(result)
            except StopAsyncIteration:
                result = None

        # объединение картинок обнаруженных лиц в единый массив
        face_tensor = torch.tensor(np.concatenate(face_lst, axis=0), device=DEVICE, dtype=DTYPE)

        # работа нейросети
        pred_val, pred_cls = net(face_tensor)

        # добавление текстовых меток о распознанных эмоциях в кадр
        # и занесение результатов работы нейросети в общий словарь
        for i, c in enumerate(pred_cls):
            fid = id_lst[i]
            cv2.putText(marked_frame_lst[fid]["marked frame"], c, faces_locs[i][:2], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            marked_frame_lst[fid]["pred values"].append(pred_val[i])
            marked_frame_lst[fid]["pred classes"].append(c)
