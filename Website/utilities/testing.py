import os
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from os import listdir
import cv2
from Website.utilities.c3d import *
import torch
from torchvision import transforms
from PIL import Image
import requests
import json

URL = 'https://www.sms4india.com/api/v1/sendCampaign'

tranfunc2 = transforms.Compose([transforms.Resize((112, 112))])

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


def pad_tensor(vec, pad, dim):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

def accident_detect(file):
	
	#video_name = 'Accident/media/output_' + file.split('.')[0] + '.avi'
	img_name = './SIH_Accident_Detection/media/output_' + file.split('.')[0] + '.png'
	print("*************************************",file)
	vidcap = cv2.VideoCapture('./SIH_Accident_Detection/media/'+file)
	print(file)
	# fps = vidcap.set(CV_CAP_PROP_FPS,10)			#set fps
	success, frame = vidcap.read()
	print(success)

	s_array = []
	i=0

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	model = C3D().to(device)
	model.load_state_dict(torch.load('/content/drive/My Drive/Colab Notebooks/SIH/utilities/checkpoints/networkc3d_train_batch_200.ckpt'))
	max_len = 209
	fullx = []
	fully = []
	counter = 0
	while success:
		img = tranfunc2(Image.fromarray(frame))
		s_array.append(np.array(img))
		success, frame = vidcap.read()
		i=i+1
		counter = counter+1
		if(i==max_len):
			s_array = np.expand_dims(s_array, 0)
			s_array_tensor = torch.tensor(s_array).type(torch.FloatTensor)
			s_array_tensor = pad_tensor(s_array_tensor, 210, 1).permute(0, 4, 1, 2, 3).to(device)
			y_pred = model(s_array_tensor)
			y_pred = y_pred[0].cpu().detach().numpy()
			print(y_pred)
			#x = range(i+1)
			#x = [counter * 0.1 for i in x]
			y = np.zeros(i+1)
			y[:i+1] = y_pred[:i+1]
			#fullx.extend(x)
			fully.extend(y)
			print(len(y))
			print(len(fully))
			s_array = []
			i = 0

	s_array = np.expand_dims(s_array, 0)
	s_array_tensor = torch.tensor(s_array).type(torch.FloatTensor)
	print(s_array_tensor.shape)
	s_array_tensor = pad_tensor(s_array_tensor, 210, 1).permute(0, 4, 1, 2, 3).to(device)

	y_pred = model(s_array_tensor)
	y_pred = y_pred[0].cpu().detach().numpy()

	print(y_pred)
	
	# threshold = 0.75
	# above_threshold = np.maximum(y_pred - threshold, 0)
	# below_threshold = np.minimum(y_pred, threshold)
	x = range(counter+1)
	x = [j * 0.1 for j in x]
	y = np.zeros(i+1)
	y[:i+1] = y_pred[:i+1]
	fullx.extend(x)
	fully.extend(y)

	#print(y)
	fig, ax = plt.subplots()
	ax.bar(fullx, fully, 1, color="r")
	#ax.bar(x, above_threshold, 0.1, color="r", bottom=below_threshold)

	# horizontal line indicating the threshold
	#ax.plot([0., 0.1*(i+5)], [threshold, threshold], "k--")

	fig.savefig(img_name)
	print(len(fully))
	flag, start_time, end_time = 0, 0.0, 0.0
	for j in range(counter):
		if fully[j] == 1 and flag == 0:
			start_time = j*0.1
			flag = 1
		elif fully[j] == 0 and flag == 1:
			end_time = (j-1)*0.1

	if(1.0 in fully):
	 	response = sendPostRequest(URL, 'DSBWWBVCNVK29IVGYKVL9V7YEJ75ZEN2', '88C294MJQNZL9IDJ', 'stage', '+918895657598', 'active-sender-id', 'There has been an accident at location: <<location provided by Maps API (to be implemented) >>' )
	 	print(response.text)

	cv2.destroyAllWindows()
	start_time = round(start_time, 2)
	end_time = round(end_time, 2)
	
	return start_time, end_time
