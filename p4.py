import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip


class Line():

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
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

def select_yellow(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([20,60,60])
    upper = np.array([38,174, 250])
    mask = cv2.inRange(hsv, lower, upper)

    return mask

def select_white(image):
    lower = np.array([202,202,202])
    upper = np.array([255,255,255])
    mask = cv2.inRange(image, lower, upper)

    return mask

def combined_threshold1(image):
  yellow = select_yellow(image)
  white = select_white(image)

  combined_binary = np.zeros_like(yellow)
  combined_binary[(yellow >= 1) | (white >= 1)] = 1

  return combined_binary

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
    M_inv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    # Return the resulting image and matrix
    return warped, M_inv

def find_lanes(binary_warped, nwindows=9, margin=100):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    
    # Create an output image to draw on and  visualize the result
    out_img_wrapped = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img_wrapped,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2) 
        cv2.rectangle(out_img_wrapped,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    left_lane.allx = nonzerox[left_lane_inds]
    left_lane.ally = nonzeroy[left_lane_inds]
    right_lane.allx = nonzerox[right_lane_inds]
    right_lane.ally = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_lane.current_fit = np.polyfit(left_lane.ally, left_lane.allx, 2)
    right_lane.current_fit = np.polyfit(right_lane.ally, right_lane.allx, 2)
    
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_lane.recent_xfitted = left_lane.current_fit[0]*ploty**2 + left_lane.current_fit[1]*ploty + left_lane.current_fit[2]
    right_lane.recent_xfitted = right_lane.current_fit[0]*ploty**2 + right_lane.current_fit[1]*ploty + right_lane.current_fit[2]

    left_lane.detected = True
    right_lane.detected  = True

    out_img_wrapped[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img_wrapped[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Define conversions in x and y from pixels space to meters (world space)
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(ploty)

    # Fitting new polynomials to x,y in world space (meters) for both lane lines
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_lane.recent_xfitted * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_lane.recent_xfitted * xm_per_pix, 2)
    # Calculate radius of curvature for each lane
    left_lane.radius_of_curvature = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_lane.radius_of_curvature = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    left_lane.line_base_pos = left_lane.recent_xfitted[-1] * xm_per_pix
    right_lane.line_base_pos = right_lane.recent_xfitted[-1] * xm_per_pix

    return out_img_wrapped

def update_lanes(binary_warped, margin=100):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_lane.current_fit[0]*(nonzeroy**2) + left_lane.current_fit[1]*nonzeroy + 
    left_lane.current_fit[2] - margin)) & (nonzerox < (left_lane.current_fit[0]*(nonzeroy**2) + 
    left_lane.current_fit[1]*nonzeroy + left_lane.current_fit[2] + margin))) 
    
    right_lane_inds = ((nonzerox > (right_lane.current_fit[0]*(nonzeroy**2) + right_lane.current_fit[1]*nonzeroy + 
    right_lane.current_fit[2] - margin)) & (nonzerox < (right_lane.current_fit[0]*(nonzeroy**2) + 
    right_lane.current_fit[1]*nonzeroy + right_lane.current_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    left_lane.allx = nonzerox[left_lane_inds]
    left_lane.ally = nonzeroy[left_lane_inds] 
    right_lane.allx = nonzerox[right_lane_inds]
    right_lane.ally = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each
    left_lane.current_fit = np.polyfit(left_lane.ally, left_lane.allx, 2)
    right_lane.current_fit = np.polyfit(right_lane.ally, right_lane.allx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    left_lane.recent_xfitted = left_lane.current_fit[0]*ploty**2 + left_lane.current_fit[1]*ploty + left_lane.current_fit[2]
    right_lane.recent_xfitted = right_lane.current_fit[0]*ploty**2 + right_lane.current_fit[1]*ploty + right_lane.current_fit[2]
    
    # Create an image to draw on and an image to show the selection window
    out_img_wrapped = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img_wrapped)
    # Color in left and right line pixels
    out_img_wrapped[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img_wrapped[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    y_eval = np.max(ploty)
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_lane.recent_xfitted * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_lane.recent_xfitted * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_lane.radius_of_curvature = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_lane.radius_of_curvature = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    
    left_lane.line_base_pos = left_lane.recent_xfitted[-1] * xm_per_pix
    right_lane.line_base_pos = right_lane.recent_xfitted[-1] * xm_per_pix
    
    return out_img_wrapped

def draw_lanes(perspective_img, Minv, undist_img):
    ploty = np.linspace(0, perspective_img.shape[0] - 1, perspective_img.shape[0])

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(perspective_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.recent_xfitted, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.recent_xfitted, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    img_size = (undist_img.shape[1], undist_img.shape[0])
    newwarp = cv2.warpPerspective(color_warp, Minv, img_size)

    # Combine the result with the original image
    out_img_unwrapped = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

    xm_per_pix = 3.7 / 700
    center = (left_lane.line_base_pos + right_lane.line_base_pos) / 2
    vehicle_pose = perspective_img.shape[1] // 2

    dx = (vehicle_pose * xm_per_pix - center)  # Positive if on right, Negative on left

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out_img_unwrapped, ('left lane curve radius = ' + str(left_lane.radius_of_curvature) + ' m'),
                (10, 100), font, 1, (255, 255, 255), 2)
    cv2.putText(out_img_unwrapped, ('right lane curve radius = ' + str(right_lane.radius_of_curvature) + ' m'),
                (10, 150), font, 1, (255, 255, 255), 2)

    cv2.putText(out_img_unwrapped, ('vehicle position in lane = ' + str(dx) + ' m'),
                (10, 200), font, 1, (255, 255, 255), 2)

    return out_img_unwrapped

def process_image(image):
    # undistort image
    undist_img = cal_undistort(image, mtx, dist)

    combined_binary = combined_threshold1(undist_img)

    img_size = image.shape
    src = np.array([[575. / 1280. * img_size[1], 460. / 720. * img_size[0]],
                    [705. / 1280. * img_size[1], 460. / 720. * img_size[0]],
                    [1127. / 1280. * img_size[1], 720. / 720. * img_size[0]],
                    [203. / 1280. * img_size[1], 720. / 720. * img_size[0]]], np.float32)
    dst = np.array([[320. / 1280. * img_size[1], 100. / 720. * img_size[0]],
                    [960. / 1280. * img_size[1], 100. / 720. * img_size[0]],
                    [960. / 1280. * img_size[1], 720. / 720. * img_size[0]],
                    [320. / 1280. * img_size[1], 720. / 720. * img_size[0]]], np.float32)
    combined_binary_top_down, Minv = unwarp(combined_binary, src, dst)

    if (left_lane.detected is False) or (right_lane.detected is False):
        output = find_lanes(combined_binary_top_down, nwindows=9, margin=100)
    else:
        output = update_lanes(combined_binary_top_down, margin=100)

    output = draw_lanes(combined_binary_top_down, Minv, undist_img)

    return output

if __name__ == "__main__":
    video = True
    mtx, dist = camera_cal('camera_cal')
    left_lane= Line()
    right_lane= Line()
    
    if video:
        input_video = 'project_video.mp4'
        # input_video = 'challenge_video.mp4'
        # input_video = 'harder_challenge_video.mp4'
        
        white_output = 'output_videos/' + input_video.split('.mp4')[0] + '_output' + '.mp4'
        
        # clip1 = VideoFileClip(input_video).subclip(41,43)
        clip1 = VideoFileClip(input_video)
        
        white_clip = clip1.fl_image(process_image)
        
        white_clip.write_videofile(white_output, audio=False)

    else:
        path = 'test_images/*.jpg'
        path = 'md_images/signs_vehicles_xygrad.jpg'
        for path in glob.glob(path):
            img = cv2.imread(path)
            output = process_image(img)

            output_image_name = path.split('\\')[-1].split('.jpg')[0] + '_output.jpg'
            cv2.imwrite('output_images/' + output_image_name, output)

            fig = plt.figure(figsize=(6, 6))
            plt.imshow(output)
            plt.title('Ouput Image')
            plt.show()

