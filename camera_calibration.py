import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob


def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist

if __name__ == "__main__":
	objp = np.zeros((6*9,3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
	
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d points in real world space
	imgpoints = [] # 2d points in image plane.
	
	# Make a list of calibration images
	images = glob.glob('camera_cal/calibration*.jpg')

	# Step through the list and search for chessboard corners
	for idx, fname in enumerate(images):
		img = cv2.imread(fname)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		# Find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

		# If found, add object points, image points
		if ret == True:
			objpoints.append(objp)
			imgpoints.append(corners)
			
			undistorted = cal_undistort(img, objpoints, imgpoints)
	
			# Draw and display the corners
			# cv2.drawChessboardCorners(img, (8,6), corners, ret)
			# #write_name = 'corners_found'+str(idx)+'.jpg'
			# #cv2.imwrite(write_name, img)
			# cv2.imshow('img', img)
			# cv2.waitKey(500)
			
			f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
			f.tight_layout()
			ax1.imshow(img)
			ax1.set_title('Original Image', fontsize=50)
			ax2.imshow(undistorted)
			ax2.set_title('Undistorted Image', fontsize=50)
			plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

plt.show()
cv2.destroyAllWindows()
