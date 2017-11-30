import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
# from moviepy.editor import VideoFileClip



def camera_cal(path, nx=9, ny=6):
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((nx*ny,3), np.float32)
	objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)
	
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.
	
	# Make a list of calibration images
	images = glob.glob(path + '/calibration*.jpg')
	
	# Step through the list and search for chessboard corners
	for fname in images:
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
	
		# If found, add object points, image points
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)
	
	# 		# Draw and display the corners
	# 		img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
	# 		cv2.imshow('img',img)
	# 		cv2.waitKey(500)
	#
	# cv2.destroyAllWindows()
	
	# Use cv2.calibrateCamera() and cv2.undistort()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	
	return mtx, dist
	
	
def cal_undistort(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh[0]) & (s_channel <= thresh[1])] = 1
    
    return binary_output

def compined_threshod(image):
	sxbinary = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20, 100))
	s_binary = hls_select(image, thresh=(170, 255))
	color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
	
	combined_binary = np.zeros_like(sxbinary)
	combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
	
	return combined_binary
	
def unwarp(img, src, dst):
    img_size = (img.shape[1], img.shape[0])
    
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped, M
	
def process_image(image):
	mtx, dist = camera_cal('camera_cal')

	combined_binary = compined_threshod(image)
	
	img_size = image.shape
	src = np.array([[575. / 1280. * img_size[1], 460. / 720. * img_size[0]],
					[705. / 1280. * img_size[1], 460. / 720. * img_size[0]],
					[1127. / 1280. * img_size[1], 720. / 720. * img_size[0]],
					[203. / 1280. * img_size[1], 720. / 720. * img_size[0]]], np.float32)
	dst = np.array([[320. / 1280. * img_size[1], 100. / 720. * img_size[0]],
                    [960. / 1280. * img_size[1], 100. / 720. * img_size[0]],
                    [960. / 1280. * img_size[1], 720. / 720. * img_size[0]],
                    [320. / 1280. * img_size[1], 720. / 720. * img_size[0]]], np.float32)
	combined_binary_top_down, perspective_M = unwarp(combined_binary, src, dst)
	
	
	output = combined_binary_top_down
	return output
	
	
	
if __name__ == "__main__":
	video = False

	if video:
		input_video = 'project_video.mp4'
		# input_video = 'challenge_video.mp4'
		# input_video = 'harder_challenge_video.mp4'
		
		white_output = 'output_videos/' + input_video.split('.mp4')[0] + '.mp4'
		
		# clip1 = VideoFileClip(input_video).subclip(0,5)
		clip1 = VideoFileClip(input_video)
		
		white_clip = clip1.fl_image(process_image)
		
		white_clip.write_videofile(white_output, audio=False)
		
	else:
		img = cv2.imread('signs_vehicles_xygrad.png')
		img = cv2.imread('test_images\straight_lines1.jpg')
		img = cv2.imread('test_images\straight_lines2.jpg')
		
		output = process_image(img)
		
		plt.figure(figsize=(15, 15))
		plt.subplot(1, 2, 1)
		plt.imshow(img)
		plt.title('Original Image')
		plt.subplot(1, 2, 2)
		plt.imshow(output)
		plt.title('Ouput Image')
		plt.show()