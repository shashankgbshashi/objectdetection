# import cv2
# from darkflow.net.build import TFNet
# print("hello")
# import matplotlib.pyplot as plt
#
# options = {
#     'model' : 'cfg/yolo.cfg',
#     'load' : 'bin/yolo.weights',
#     'thresold' : 0.3
# }
#
# tfnet = TFNet(options)
# #print(tfnet)
#
# image = cv2.imread('cat.jpg')
#
# result = tfnet.return_predict(image)
#
# #print(result[0])
#
# for results in result:
#     if results['confidence'] > 0.5:
#         #t1 =() results['topleft']['x']
#         image = cv2.rectangle(image,(results['topleft']['x'],results['topleft']['y']),(results['bottomright']['x'],results['bottomright']['y']),(0,255,0),2)
#         cv2.putText(image,results['label'],(results['bottomright']['x'],results['bottomright']['y']),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1,cv2.LINE_AA)
#         cv2.imshow("shashank",image)
#         k = cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         #print(results)
#
#
#
#
#
#
#
#



import cv2
from darkflow.net.build import TFNet

options = {
    'model' : 'cfg/yolo.cfg',
    'load' : 'bin/yolo.weights',
    'thresold' : 0.3
}

tfnet = TFNet(options)

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    _,frame = cap.read()
    results = tfnet.return_predict(frame)
    for box in results:
        if box['confidence'] > 0.5:
            t1 = (box['topleft']['x'],box['topleft']['y'])
            t2 = (box['bottomright']['x'],box['bottomright']['y'])
            cv2.rectangle(frame,t1,t2,(255,0,0),2,cv2.LINE_AA)
            cv2.putText(frame,box['label'],t1,cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),5,cv2.LINE_AA)

    cv2.imshow('shashank',frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
