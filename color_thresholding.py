import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
	
    return binary_output
    

if __name__ == "__main__":
	image = mpimg.imread('signs_vehicles_xygrad.png')
	
	thresh_s=(20, 50)
	
	hls_binary = hls_select(image, thresh=thresh_s)
	
	# Plot the result
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
	f.tight_layout()
	ax1.imshow(image)
	ax2.imshow(hls_binary, cmap='gray')
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
	plt.show()