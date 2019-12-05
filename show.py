import argparse
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
import os
import errno



def cal_dist(params):
	a=np.asarray(params[0])
	b=np.asarray(params[1])
	#ForkedPdb().set_trace()
	#print(type(a))
	dist=np.linalg.norm(a-b)
	if abs(dist) < 75 :
		return True
	else:
		return False
	 
def find_if_close(cnt1,cnt2):
	row1,row2 = cnt1.shape[0],cnt2.shape[0]
  #  for i in xrange(row1):
  #      for j in xrange(row2):
	paramlist=list(itertools.product(cnt1,cnt2))
	#import pdb;pdb.set_trace()
	#ForkedPdb().set_trace()
	#dist = np.linalg.norm(cnt1[i]-cnt2[j])
	print('Starting distance calculation for two contours')
	pool = multiprocessing.Pool(60)
	#dist=1000
	#res=[]
	res=pool.map(cal_dist,paramlist)
	pool.close() 
	#print('This pair is done')
	return any(res)
	#ForkedPdb().set_trace()
def show_image(al,path_mask,dest_name):
	for id in tqdm(al):
	#for id in tqdm(range(1,11)):
		
		#second=al[id][0] 
		src=path_mask+'/'+str(id)+'.pickle'
		img=al[id][0]
		cl_img=al[id][1]
		
		with open(src,'rb')as fp: 
			mask_l=pickle.load(fp) 
		#import pdb;pdb.set_trace()
		unified,contours=mask_l[0],mask_l[1]
		mask_mid=np.ones(img.shape, np.uint8)
		mask = np.ones(img.shape, np.uint8)
		cv2.fillPoly(mask, unified[0:1], color=(0))
		cv2.fillPoly(mask_mid, contours[0:18], color=(0))
		#plt.figure()
		
		plt.figure(figsize=(15,15))
		plt.subplots_adjust(wspace=0.01,hspace=0.01)	
		plt.subplot(231)
		plt.imshow(img,cmap='gray') 
		
		#plt.imshow(np.round(np.expand_dims(img,axis=-1) * cl_img).astype('uint8'), cmap='jet' )
		plt.title('Output From MNET')
		plt.axis('off')
		plt.subplot(232)
		plt.imshow(mask_mid,cmap='binary')
		#plt.imshow(np.round(np.expand_dims(mask,axis=-1) * cl_img).astype('uint8'), cmap='jet' )
		plt.axis('off')

		plt.title('Output From Morphological Operations')

		plt.subplot(233)
		plt.imshow(mask,cmap='binary')
		plt.axis('off')
		#plt.imshow(np.round(np.expand_dims(mask,axis=-1) * cl_img).astype('uint8'), cmap='jet' )

		plt.title('Segmented Image')
		
		plt.subplot(234)
		plt.axis('off')
		#plt.imshow(mask,cmap='binary')
		plt.imshow(cl_img.astype('uint8'))
		plt.title('Original Image')
	
		plt.subplot(235)
		plt.axis('off')
		#plt.imshow(mask,cmap='binary')
		plt.imshow(np.round(np.expand_dims(mask,axis=-1) * cl_img).astype('uint8'), cmap='jet' )
		plt.title('Masked Image')
		
		file_name=dest_name+'/'+str(id)+'.jpg'
	#	file_name='res_raph/'+str(i)+'.mat'
		plt.savefig(file_name)
		#print('This image is done')	
	#	scipy.io.savemat(file_name, {'binary':mask})
		#import pdb;pdb.set_trace()

		plt.show()
if __name__ == '__main__':
	parser=argparse.ArgumentParser()
	parser.add_argument('-l','--location',nargs='+',required=True,help='Path to the pickle file which contains ManTranet output')
	parser.add_argument('-s','--source',nargs='+',required=True,help='Path to the folder to store all the filtered masks')
	parser.add_argument('-d','--destination',nargs='+',required=True,help='Name of the folder to store all the result image')
	args=parser.parse_args()
	path=args.location[0]+'.pickle'
	path_mask=args.source[0]
	folder_name=args.destination[0]
        #num_cores=args.num_cores
	with open(path,'rb')as fp: 
		al=pickle.load(fp) 
#	with open(path_mask,'rb')as fp: 
#		src=pickle.load(fp) 
	try:
		os.mkdir(folder_name)
	except OSError as exc:
		if exc.errno != errno.EEXIST:
			raise
		pass
	#import pdb;pdb.set_trace()
	show_image(al,path_mask,folder_name)
	
import pdb;pdb.set_trace()


