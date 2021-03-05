import cv2
import numpy as np
import os, sys


imgs = []
path = "/Users/jasonli/Desktop/BU/Junior/Spring2021/CS791/herbarium/Herbarium_Project/assignments/recognizeme/"


# names = ["temp55.jpg", "temp56.jpg", "temp57.jpg", "temp58.jpg", "temp59.jpg"]
# for img_name in os.listdir(path):

# 	if img_name not in names:
# 		continue

# 	img = os.path.join(path, img_name)
# 	img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

# 	imgs.append(img)

# imposed = np.zeros_like(imgs[0])
# h,w = imposed.shape
# for img in imgs:
# 	imposed += img
# imposed = imposed / len(imgs)

# cv2.imwrite(path+"asdf.jpg", imposed)
# print(imposed.sum() / (h*w))

# # imposed = np.uint8(imposed)
# imposed = imposed.astype(np.uint8)

imposed = cv2.imread(path+"temp2.jpg", cv2.IMREAD_GRAYSCALE)
imposed = imposed.astype(np.uint8)
edged = cv2.Canny(imposed, 30, 200) 
x,y,w,h = cv2.boundingRect(edged)

# laplacian = cv2.Laplacian(imposed,cv2.CV_64F)
temp = [1 if p > 0 else 0 for x in imposed for p in x]
print(sum(temp) / (w*h))
# print((w*h)/ (imposed.shape[0]*imposed.shape[1]))
# # if w*h > img
edged = cv2.rectangle(imposed, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the output
# cv2.imshow('img', edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print(boundrec)

cv2.imwrite(path+"t.jpg", edged)
quit()

contours, hierarchy = cv2.findContours(imposed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contour_output = edged.copy()
contour_output = np.zeros(np.shape(imposed), dtype='uint8') # new blank canvas
print(len(contours))
if len(contours) != 0:
	# contour = max(contours, key=lambda x: cv2.contourArea(x))
	# boundrec = cv2.boundingRect(contour)
	cv2.drawContours(contour_output, contours, -1, (255,0,0), 2)
	# cv2.rectangle(contour_output, boundrec, (0,255,0))

cont_area = sum([cv2.contourArea(contour) for contour in contours])
print(cont_area / (w*h))# (imposed.shape[0]*imposed.shape[1]))
# cv2.namedWindow("Contours2", cv2.WINDOW_AUTOSIZE);
# cv2.imshow("Contours2", contour_output);
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.waitKey(-1);
cv2.imwrite(path+"cont.jpg", contour_output)
