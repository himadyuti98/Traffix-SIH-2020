import cv2
import numpy as np
import glob
import os
 
frames = os.listdir('../frames')

for frame in frames:
	img_array = []
	files = os.listdir('../frames/{}/images'.format(frame))
	files.sort()
	print(files)
	for filename in files:
	    img = cv2.imread('../frames/{}/images/'.format(frame) + filename)
	    height, width, layers = img.shape
	    size = (width,height)
	    img_array.append(img)
	 
	 
	out = cv2.VideoWriter('../frames/{}/video.avi'.format(frame),cv2.VideoWriter_fourcc(*'MJPG'), 10, size)
	 
	for i in range(len(img_array)):
	    out.write(img_array[i])
	out.release()

	os.system("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = '../frames/{}/video.avi'.format(frame), output = '../frames/{}/video'.format(frame)))