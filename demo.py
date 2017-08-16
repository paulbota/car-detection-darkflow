from darkflow.net.build import TFNet
import cv2
import glob
from os.path import basename

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

globs = glob.glob("./test/*.jpg")

for g in globs:
    imgcv = cv2.imread(g)
    result = tfnet.return_predict(imgcv)
    print(result)
    count = 0
    for r in result:
        if r['label'] == 'car' or r['label'] == 'truck':
            count += 1
        xmin = r['topleft']['x']
        ymin = r['topleft']['y']
        xmax = r['bottomright']['x']
        ymax = r['bottomright']['y']
        cv2.rectangle(imgcv, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
        cv2.putText(imgcv, r['label'], (int(xmin), int(ymin)), 2, 0.5, (0, 0, 255))
    print(count)
    cv2.imwrite('results/' + basename(g), imgcv)

