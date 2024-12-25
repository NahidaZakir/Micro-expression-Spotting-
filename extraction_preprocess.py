import numpy as np
import pandas as pd
import cv2
import dlib

def pol2cart(rho, phi): #Convert polar coordinates to cartesian coordinates for computation of optical strain
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def computeStrain(u, v):
    u_x= u - pd.DataFrame(u).shift(-1, axis=1)
    v_y= v - pd.DataFrame(v).shift(-1, axis=0)
    u_y= u - pd.DataFrame(u).shift(-1, axis=0)
    v_x= v - pd.DataFrame(v).shift(-1, axis=1)
    os = np.array(np.sqrt(u_x**2 + v_y**2 + 1/2 * (u_y+v_x)**2).ffill().ffill())
    return os

#This code passes final_images 
def extract_hog_top(images, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2,2), selected_frames=3, eps=1e-7):
    # Assume 'images' is a list of image arrays already loaded (as passed from final_images)
    
    video_length = len(images)
    print(video_length)
    middle_frame = video_length // 2
    selected_3frames = [0, middle_frame, video_length - 1]
    
    img_seq = []  # list to store the selected images
    for i in selected_3frames:
        img = cv2.resize(images[i], (128, 64))  # Resize each frame
        img_seq.append(img)

    # Convert list to NumPy array
    img_seq = np.array(img_seq)

    # Transpose the img_seq to extract features from XT and YT planes
    xt = np.transpose(img_seq, (1, 0, 2, 3))
    yt = np.transpose(img_seq, (2, 0, 1, 3))

    # Initialize histograms
    hist_xy_plane = np.empty(0, dtype=np.float32)
    hist_xt_plane = np.empty(0, dtype=np.float32)
    hist_yt_plane = np.empty(0, dtype=np.float32)

    n_frames_xy_plane = img_seq.shape[0]
    n_frames_xt_plane = xt.shape[0]
    n_frames_yt_plane = yt.shape[0]

    # Compute HOG features for XY plane
    for i in range(n_frames_xy_plane):
        hist_xy, hog_xy = hog(img_seq[i], orientations=orientations, pixels_per_cell=pixels_per_cell,
                              cells_per_block=cells_per_block, block_norm='L2', visualize=True, channel_axis=-1)
        hist_xy_plane = np.concatenate((hist_xy_plane, hist_xy))

    # Compute HOG features for XT plane
    for i in range(n_frames_xt_plane):
        hist_xt, hog_xt = hog(xt[i], orientations=orientations, pixels_per_cell=(3, 8),
                              cells_per_block=(2, 2), block_norm='L2', visualize=True, channel_axis=-1)
        hist_xt_plane = np.concatenate((hist_xt_plane, hist_xt))

    # Compute HOG features for YT plane
    for i in range(n_frames_yt_plane):
        hist_yt, hog_yt = hog(yt[i], orientations=orientations, pixels_per_cell=(3, 8),
                              cells_per_block=cells_per_block, block_norm='L2', visualize=True, channel_axis=-1)
        hist_yt_plane = np.concatenate((hist_yt_plane, hist_yt))

    # Concatenate features from all planes
    hog_top = np.concatenate((hist_xy_plane, hist_xt_plane, hist_yt_plane))
    hog_top = hog_top / np.linalg.norm(hog_top + eps)  # Avoid division by zero

    return hog_top



def extract_preprocess(final_images, k):
    #Path to the pre-trained dlib shape predictor model.
    predictor_model = "Utils\\shape_predictor_68_face_landmarks.dat"
    #Initialized dlib's face detector.
    face_detector = dlib.get_frontal_face_detector()
    # Initialized dlib's shape predictor for facial landmarks.
    face_pose_predictor = dlib.shape_predictor(predictor_model)
    #An empty list to store the processed data for all videos.
    dataset = []
    for video in range(len(final_images)):
      OFF_video = []
      for img_count in range(final_images[video].shape[0]-k):
        #Taking only images separted by k interval frames
        img1 = final_images[video][img_count]
        img2 = final_images[video][img_count+k]
        #Detecting Facial Landmarks, For the first frame, it attempts to detect a face. 
        #If no face is detected, it continues to the next frame until a face is found.
        #Once a face is detected, it uses face_pose_predictor to get the landmarks.
        
        if (img_count==0):
            reference_img = img1
            detect = face_detector(reference_img,1)
            next_img=0 #Loop through the frames until all the landmark is detected
            while (len(detect)==0):
                next_img+=1
                reference_img = final_images[video][img_count+next_img]
                detect = face_detector(reference_img,1)
            shape = face_pose_predictor(reference_img,detect[0])
            
            #Left Eye
            x11=max(shape.part(36).x - 15, 0)
            y11=shape.part(36).y 
            x12=shape.part(37).x 
            y12=max(shape.part(37).y - 15, 0)
            x13=shape.part(38).x 
            y13=max(shape.part(38).y - 15, 0)
            x14=min(shape.part(39).x + 15, 128)
            y14=shape.part(39).y 
            x15=shape.part(40).x 
            y15=min(shape.part(40).y + 15, 128)
            x16=shape.part(41).x 
            y16=min(shape.part(41).y + 15, 128)
            
            #Right Eye
            x21=max(shape.part(42).x - 15, 0)
            y21=shape.part(42).y 
            x22=shape.part(43).x 
            y22=max(shape.part(43).y - 15, 0)
            x23=shape.part(44).x 
            y23=max(shape.part(44).y - 15, 0)
            x24=min(shape.part(45).x + 15, 128)
            y24=shape.part(45).y 
            x25=shape.part(46).x 
            y25=min(shape.part(46).y + 15, 128)
            x26=shape.part(47).x 
            y26=min(shape.part(47).y + 15, 128)
            
            #ROI 1 (Left Eyebrow)
            x31=max(shape.part(17).x - 12, 0)
            y32=max(shape.part(19).y - 12, 0)
            x33=min(shape.part(21).x + 12, 128)
            y34=min(shape.part(41).y + 12, 128)
            
            #ROI 2 (Right Eyebrow)
            x41=max(shape.part(22).x - 12, 0)
            y42=max(shape.part(24).y - 12, 0)
            x43=min(shape.part(26).x + 12, 128)
            y44=min(shape.part(46).y + 12, 128)
            
            #ROI 3 #Mouth
            x51=max(shape.part(60).x - 12, 0)
            y52=max(shape.part(50).y - 12, 0)
            x53=min(shape.part(64).x + 12, 128)
            y54=min(shape.part(57).y + 12, 128)
            
            #Nose landmark
            x61=shape.part(28).x
            y61=shape.part(28).y
    
        #Compute Optical Flow Features
        #optical_flow = cv2.DualTVL1OpticalFlow_create() #Depends on cv2 version
        optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
        #3D array ouput of the optical flow, each element contains a (u,v) array where u means horizontal displacement at position
        #(x,y) and v means vertical displacement at position (x,y)
        flow = optical_flow.calc(img1, img2, None)
        
        #This converts the Cartesian coordinates (u, v) of the flow vectors to polar coordinates (magnitude, angle).
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1]) 
        #This converts the polar coordinates back to Cartesian coordinates. This step is based on the assumption that pol2cart is defined elsewhere.
        u, v = pol2cart(magnitude, angle)
        
        #This computes the strain (a measure of deformation) from the displacement vectors (u, v). The specifics of computeStrain 
        #depend on its implementation, but it typically quantifies how much the shape of the object has changed between img1 and img2.
        os = computeStrain(u, v)
                
        #Features Concatenation into 128x128x3
        final = np.zeros((128, 128, 3))
        final[:,:,0] = u
        final[:,:,1] = v
        final[:,:,2] = os
        
        #Remove global head movement by minus nose region
        final[:, :, 0] = abs(final[:, :, 0] - final[y61-5:y61+6, x61-5:x61+6, 0].mean())
        final[:, :, 1] = abs(final[:, :, 1] - final[y61-5:y61+6, x61-5:x61+6, 1].mean())
        final[:, :, 2] = final[:, :, 2] - final[y61-5:y61+6, x61-5:x61+6, 2].mean()
        
        #Eye masking
        left_eye = [(x11, y11), (x12, y12), (x13, y13), (x14, y14), (x15, y15), (x16, y16)]
        right_eye = [(x21, y21), (x22, y22), (x23, y23), (x24, y24), (x25, y25), (x26, y26)]
        cv2.fillPoly(final, [np.array(left_eye)], 0)
        cv2.fillPoly(final, [np.array(right_eye)], 0)
        
        #Extracts the regions of interest (eyebrows and mouth) and resamples them to a fixed size 
        #Concatenates the resampled regions into a final image (42x42x3).
        #ROI Selection -> Image resampling into 42x22x3
        final_image = np.zeros((42, 42, 3))
        final_image[:21, :, :] = cv2.resize(final[min(y32, y42) : max(y34, y44), x31:x43, :], (42, 21))
        final_image[21:42, :, :] = cv2.resize(final[y52:y54, x51:x53, :], (42, 21))
        OFF_video.append(final_image)
        
      dataset.append(OFF_video)
      print('Video', video, 'Done')
    print('All Done')
    return dataset

    

