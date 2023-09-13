from typing import Any
import cv2
import torch
import numpy as np
from nn import NN 

NUM_FRAME = 1000

CLASSIFIER_PATH = "./models/haarcascade_frontalface_default.xml"
NN_W_PATH = "./models/FER_trained_model.pt"

EMOTION_DICT = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness', 4: 'anger', 5: 'disguest', 6: 'fear'}

DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.set_num_threads(1)

print(f"Device: {DEVICE}")
print(f"Number of threads: {torch.get_num_threads()}")


class FERNN_model:
    def __init__(self, model_path, em_dict, device=None, dtype=None):
        self.model = NN(device=device, dtype=dtype)
        self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
        self.emotion_dict = em_dict

    def __call__(self, img):
        with torch.no_grad():
            self.model.eval()
            log_ps = self.model(img).cpu()
            ps = torch.exp(log_ps)
            top_v, top_c = ps.topk(1, dim=1)
            return top_v.numpy(), self.emotion_dict[int(top_c.numpy())]


class FaceDetectionGenerator:
    def __init__(self, id: int, v_path: str, s_path: str, c_path: str, max_num_frame: int): # s_path: str, c_path: str, fps: int, frame_size: tuple | list, max_num_frame: int):
        self.id = id
        self.net = FERNN_model(NN_W_PATH, EMOTION_DICT, DEVICE, DTYPE)
        self.vc = cv2.VideoCapture(v_path)
        fw = int(self.vc.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
        fh = int(self.vc.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
        vc_fps = self.vc.get(cv2.CAP_PROP_FPS)
        print(f"VC {id} - FPS = {vc_fps}")
        # fourcc = 0
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # self.vw = cv2.VideoWriter(s_path, 0, 10.0, (fw, fh))
        self.vw = cv2.VideoWriter(s_path, fourcc, vc_fps, (fw, fh))
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
            pv_lst = []
            pc_lst = []
            for (x, y, w, h) in faces_loc:
                # добавление прямоугольника рамки в картинку
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                # извлечение изображения лица по рамке,
                # уменьшение размера картинки до 48*48 и нормализация
                face = cv2.resize(gray[y:y + h, x:x + w], (48, 48))[None,None,:,:] / 256.0

                # работа нейросети
                pred_val, pred_cls = self.net(torch.tensor(face, device=DEVICE, dtype=DTYPE))

                cv2.putText(frame, pred_cls, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

                face_lst.append(face)
                pv_lst.append(pred_val)
                pc_lst.append(pred_cls)

            # if self.vw.isOpened():
            self.vw.write(frame)

            yield {
                "id":self.id, 
                "frame num":frame_num,
                "frame time":frame_time,
                "marked frame":frame, 
                "detected faces":face_lst, 
                "faces loc":faces_loc,
                "pred values":pv_lst,
                "pred classes":pc_lst
            }
            
            rval, frame = self.vc.read()
                
            if frame_num == self.max_num_frame:
                break
        
        self.vc.release()
        self.vw.release()


async def main(ulf_path_list: list, pf_path_list: list) -> None:

    fd_lst = [
        FaceDetectionGenerator(i, p, pf_path_list[i], CLASSIFIER_PATH, NUM_FRAME)()
        for i, p in enumerate(ulf_path_list)
    ]
    
    for _ in range(NUM_FRAME):
        marked_frame_lst = []
        for g in fd_lst:
            try:
                result = await anext(g)
                marked_frame_lst.append(result)
            except StopAsyncIteration:
                result = None
