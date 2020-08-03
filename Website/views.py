from django.shortcuts import render
from .models import *
from .utilities import testing
import os
import cv2

# get request
def sendPostRequest(reqUrl, apiKey, secretKey, useType, phoneNo, senderId, textMessage):
  req_params = {
  'apikey':apiKey,
  'secret':secretKey,
  'usetype':useType,
  'phone': phoneNo,
  'message':textMessage,
  'senderid':senderId
  }
  return requests.post(reqUrl, req_params)


def index(request):
	if request.method == 'GET':
		return render(request, 'index.html', {'flag': 1})
	else:
		# i = 0
		# img_array = []
		# vid = cv2.VideoCapture(0) 
		# while(i<210):
		# 	ret, frame = vid.read() 
		# 	img_array.append(frame)

		# invid = cv2.VideoWriter('./media/video.avi'.format(frame),cv2.VideoWriter_fourcc(*'MJPG'), 10, size) 
		invid = request.FILES['invid']

		print(invid)

		vid = Website_vid()
		vid.inputVid = invid
		vid.save()

		oname = str(vid.inputVid)
		print("#########")

		start_time, end_time = testing.accident_detect(str(vid.inputVid))
		
		vid.outputImg = 'output_' + oname.split('.')[0] + '.png'
		
		#nm = 'output_' + oname.split('.')[0]
		
		#os.system("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = 'RainRemove/media/' + str(vid.outputVid), output = 'RainRemove/media/' + nm))

		#vid.outputVid = nm + '.mp4'
		vid.save()
		

		return render(request, 'index.html', {'flag': 2, 'vid': vid, 'start_time': start_time, 'end_time': end_time})
