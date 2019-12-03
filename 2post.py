import pickle
#from scipy import fftpack
from scipy import fftpack, ndimage 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imutils
import scipy.io
from PIL import Image
from past.builtins import xrange
import itertools
import multiprocessing
import pdb
import sys
import imutils
from tqdm import tqdm
with open('res_test.pickle','rb')as fp:
	al=pickle.load(fp) 
#with open('res.pickle','rb')as fp:
#	al=pickle.load(fp) 
class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def cal_dist(params):
	a=np.asarray(params[0])
	b=np.asarray(params[1])
	#ForkedPdb().set_trace()
	#print(type(a))
	dist=np.linalg.norm(a-b)
	if abs(dist) < 150 :
		return True
	else:
		return False
	 
def find_if_close(cnt1,cnt2):
	row1,row2 = cnt1.shape[0],cnt2.shape[0]
	paramlist=list(itertools.product(cnt1,cnt2))
	print('Starting distance calculation for two contours')
	num_core=80
	max_iter=len(paramlist)
	index=0
	dist=False
	pool = multiprocessing.Pool(num_core)
	res=pool.map(cal_dist,paramlist)
	pool.close() 
#	for k in range(max_iter):   
#		pool = multiprocessing.Pool(num_core)
#		res=pool.map(cal_dist,paramlist[index:index+num_core])
#		index=index+num_core
#		pool.close() 
#		if any(res)==True:
#			dist=True
#			break;
#		else:
#			dist=False
	
	return any(res)
	#ForkedPdb().set_trace()
for id in tqdm(range(1,11)):
	second=al[id][0] 
	src='test_images/'+str(id)+'.jpg'
	#src='all/'+str(id)+'.jpg'

	cl_img=np.array(Image.open(src),'uint8')
	img=np.array(second*255,dtype='uint8')
	im=img
	img_f=np.tile(img[:, :, None], [1, 1, 3])

	morph = im.copy()

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
	morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
	#morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

	# take morphological gradient
	gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)

	# split the gradient image into channels
	image_channels = gradient_image
	#image_channels = morph

	#channel_height, channel_width, _ = image_channels[0].shape

	# apply Otsu threshold to each chann

	_, image_channels = cv2.threshold(~image_channels, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
#	image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))
#
#	# merge the channels
#	image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)
	contours_p, heirarchy = cv2.findContours(~image_channels, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(contours_p, key = cv2.contourArea, reverse = True)
	#for cnt in cnts:
	#import pdb;pdb.set_trace()
	contours=cnts[0:25]
	LENGTH = len(contours)
	status = np.zeros((LENGTH,1))
	for i,cnt1 in enumerate(contours):
		x = i    
		if i != LENGTH-1:
			for j,cnt2 in enumerate(contours[i+1:]):
				x = x+1
				dist = find_if_close(cnt1,cnt2)
				print('This Pair is Done')				
				if dist == True:
					val = min(status[i],status[x])
					status[x] = status[i] = val
				else:
					if status[x]==status[i]:
						status[x] = i+1
	print('All distances are measured; its time for drawing combined regions')
	#import pdb;pdb.set_trace()
	unified = []
	maximum = int(status.max())+1
	for i in xrange(maximum):
		pos = np.where(status==i)[0]
		if pos.size != 0:
			cont = np.vstack(contours[i] for i in pos)
			hull = cv2.convexHull(cont)
			unified.append(hull)
	print('Combining regions is done')
	meta='metadatatest_f/'+str(id)+'.pickle'
	with open(meta,'wb')as fp:pickle.dump(unified,fp,protocol=pickle.HIGHEST_PROTOCOL)
	print('This image is done')	
	
import pdb;pdb.set_trace()


