from utils.utils import *
from utils.datasets import *


def detector(frame, model, device, half=True):
    img_size = 640
    img0 = frame
    img = letterbox(img0, new_shape=img_size)[0]

    img = img[:, :, :].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.1)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                info_list = []
                for *xyxy, conf, cls in det:
                    xyxy = torch.tensor(xyxy).view(-1).tolist()
                    info = [xyxy[0], xyxy[1], xyxy[2], xyxy[3], int(cls)]
                    info_list.append(info)
                return info_list
            else:
                return None