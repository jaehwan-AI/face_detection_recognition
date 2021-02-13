import cv2
import torch
from models import efficientnet

def draw_bbox(frame, boxes, probs):
    '''
    Draw bounding box and probs
    '''
    for box, prob in zip(boxes, probs):
        # draw rectangle on frame
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,0,255), thickness=2)

        # show probability
        cv2.putText(frame, str(prob), (box[2], box[3]),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
    return frame

def detect_rois(boxes):
    '''
    return rois as a list
    '''
    rois = list()
    for box in boxes:
        roi = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
        rois.append(roi)
    return rois

def def_model(label, device):
    '''
    define model
    '''
    if label == 'gender':
        model = efficientnet.efficientnet_b0(num_classes=2)
        model.load_state_dict(torch.load('./trained_model/gender_model_best.pth.tar', map_location=device)['state_dict'])
    elif label == 'gaze':
        model = efficientnet.efficientnet_b0(num_classes=5)
        model.load_state_dict(torch.load('./trained_model/gaze_model_best.pth.tar', map_location=device)['state_dict'])
    elif label == 'emotion':
        model = efficientnet.efficientnet_b0(num_classes=5)
        model.load_state_dict(torch.load('./trained_model/emotion_model_best.pth.tar', map_location=device)['state_dict'])
    elif label == 'multimodal':
        model = efficientnet.efficientnet_b0(num_classes=2)
        model.load_state_dict(torch.load('./trained_model/multimodal_model_best.pth.tar', map_location=device)['state_dict'])
    return model

def predict(face, model, device):
    '''
    data preprocessing & predict label
    '''
    # preprocessing image data
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (224,224))
    img_tensor = torch.tensor(face, dtype=torch.float32)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    img_tensor = torch.unsqueeze(img_tensor, 0)

    # predict label
    model.to(device)
    model.eval()
    with torch.no_grad():
        data = img_tensor.to(device)
        out = model(data)
    return int(out.argmax(dim=1)[0])
