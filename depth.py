import cv2
import torch
import matplotlib.pyplot as plt
# from midas.dpt_depth imp
#Download the midas from 
# midas=torch.hub.load('intel-isl/MiDaS','MiDaS_small')
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to('cpu')
midas.eval()
#Input transformational 
transforms=torch.hub.load('intel-isl/MiDaS','transforms')
transform=transforms.small_transform

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    imgbatch=transform(img).to('cpu')
    # Make a prediction
    with torch.no_grad():
        prediction  = midas(imgbatch)
        prediction=torch.nn.functional.interpolate(prediction.unsqueeze(1),size=img.shape[:2],mode='bicubic',align_corners=False).squeeze()
        output=prediction.cpu().numpy()
        # print(prediction)
    plt.imshow(output)
    cv2.imshow('CV2Frame',frame)
    plt.pause(0.00001)
    # cv2.waitKey(1)
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
plt.show()



