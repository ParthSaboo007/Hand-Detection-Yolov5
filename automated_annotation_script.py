import argparse
import glob
import os
import os.path

import cv2

from yolo import YOLO

save_path = '../../label_directory_name'   # location of folder where labels will be stored

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', default="images", help='Path to images or image file')
ap.add_argument('-n', '--network', default="normal", choices=["normal", "tiny", "prn", "v4-tiny"],
                help='Network Type')
ap.add_argument('-d', '--device', default=0, help='Device to use')
ap.add_argument('-s', '--size', default=416, help='Size for yolo')
ap.add_argument('-c', '--confidence', default=0.25, help='Confidence for yolo')
args = ap.parse_args()

if args.network == "normal":
    print("loading yolo...")
    yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
elif args.network == "prn":
    print("loading yolo-tiny-prn...")
    yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
elif args.network == "v4-tiny":
    print("loading yolov4-tiny-prn...")
    yolo = YOLO("models/cross-hands-yolov4-tiny.cfg", "models/cross-hands-yolov4-tiny.weights", ["hand"])
else:
    print("loading yolo-tiny...")
    yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])

yolo.size = int(args.size)
yolo.confidence = float(args.confidence)

print("extracting tags for each image...")
if args.images.endswith(".txt"):
    with open(args.images, "r") as myfile:
        lines = myfile.readlines()
        files = map(lambda x: os.path.join(os.path.dirname(args.images), x.strip()), lines)
else:
    files = sorted(glob.glob("%s/*.jpg" % args.images))

conf_sum = 0
detection_count = 0

for file in files:
    print(file)
    mat = cv2.imread(file)

    width, height, inference_time, results = yolo.inference(mat)

    print("%s in %s seconds: %s classes found!" %
          (os.path.basename(file), round(inference_time, 2), len(results)))

    output = []

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 720, 640)

    # declaring various lists
    #lx, ly, lw, lh, lc = ([],) * 5
    handclass = 0
    for detection in results:
        id, name, confidence, x, y, w, h = detection
        # x -> top left x-coordinate
        # y - > to-left y-coordinate
        # w -> bounding box width
        # h -> bounding box height
        # wimg -> width of img
        # himg -> height of image

        cx = x + (w / 2)
        cy = y + (h / 2)

        conf_sum += confidence
        detection_count += 1

        # draw a bounding box rectangle and label on the image
        color = (185, 15, 10)
        cv2.rectangle(mat, (x, y), (x + w, y + h), color, 10) # thickness of rectangle = 10px
        text = "%s (%s)" % (name, round(confidence, 2))
        cv2.putText(mat, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    5, color, 10)

        print("%s with %s confidence" % (name, round(confidence, 2)))



        # cv2.imwrite("export.jpg", mat)

    if len(results)>0:
        wimg = 1280
        himg = 736
        w1 = w / wimg
        h1 = h / himg
        ww1 = results[0][5]
        hh1 = results[0][6]
        xx1 = (results[0][3] + (ww1 / 2)) / wimg
        yy1 = (results[0][4] + (hh1 / 2)) / himg
        wf1 = ww1 / wimg
        hf1 = hh1 / himg
        fname1 = os.path.basename(file)
        fname2 = os.path.join(save_path,fname1)
        file_name = fname2.replace('jpg', '') + "txt"  # saving the label text file of the image

        line1 = str(handclass) + " " + str(xx1) + " " + str(yy1) + " " + str(wf1) + " " + str(hf1)
        with open(file_name, 'w+') as f:
            f.write(line1)

    print('results', results)

    #print(results[1][1])

# to show the annotated images simultaneously
"""
    # show the output image
    cv2.imshow('image', mat)
    print(mat.shape)
    cv2.waitKey(0)
"""

print("AVG Confidence: %s Count: %s" % (round(conf_sum / detection_count, 2), detection_count))
cv2.destroyAllWindows()
