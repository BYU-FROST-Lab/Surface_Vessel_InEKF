import numpy as np
import cv2
from matplotlib import pyplot as plt
from src.core.measurements import HorizonObservation
import time

Radius_Earth = 6378137.0  # in meters



class ImageProcessor:
    def __init__(self, mtx, dist, img_width, img_height, camera_height):
        self.mtx = mtx
        self.dist = dist
        self.img_width = img_width
        self.img_height = img_height
        self.new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (img_width, img_height), 1, (img_width, img_height))
        self.camera_height = camera_height

        self.lsd = cv2.createLineSegmentDetector(0)

    # Takes in an image, returns roll and pitch measurement in radians
    def lsd_image_callback(self, image, timestamp_sec=0, timestamp_nsec=0, debug=False):
        # if img_width == -1: # Wait for camera info message
        #     return
        # Function to calculate line length
        def line_length(line):
            x1, y1, x2, y2 = line[0]
            return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # convert image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (self.img_width, self.img_height), 1, (self.img_width, self.img_height))
        undistorted = cv2.undistort(image, self.mtx, self.dist, None, newcameramtx)
        lines = self.lsd.detect(undistorted)[0]

        ####################### Debug block: Draw all lines on the image #######################
        if debug:
            debug_image = undistorted.copy()
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = map(int, line[0])
                    cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw in green

            # Display the debug image
            plt.figure()
            plt.imshow(debug_image)
            plt.show()
        ####################### End Debug block #######################

        # Filter Lengths:
        lengths = [line_length(line) for line in lines]
        # Define a minimum length threshold
        min_length_threshold = self.img_width/3  # Threshold in pixels

        # Filter lines by length
        filtered_lines = [line for line, length in zip(lines, lengths) if length >= min_length_threshold]

        # remove vertical lines
        filtered_lines = [line for line in filtered_lines if abs(line[0][0] - line[0][2]) > 2]

        # remove horizontal lines near the top and bottom of the image
        filter_buffer = self.img_height/10
        filtered_lines = [line for line in filtered_lines if line[0][1] > filter_buffer and line[0][3] > filter_buffer]
        filtered_lines = [line for line in filtered_lines if line[0][1] < self.img_height - filter_buffer and line[0][3] < self.img_height - filter_buffer]
        
        if len(filtered_lines) > 1:
            lengths = [line_length(line) for line in filtered_lines]
            filtered_lines = [line for _, line in sorted(zip(lengths, filtered_lines), key=lambda x: x[0], reverse=False)]

        est_angles = []
        # for line in filtered_lines: # sort lines such that x1 < x2
        try:
            line = filtered_lines[-1]  # Take the longest line only
            x1, y1, x2, y2 = line[0]
            # sort such that x1 < x2
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
            est_angles.append(np.arctan2(y2-y1, x2-x1))
            est_angle = np.mean(est_angles) * -1 # Flipping sign to match the coordinate frame

            # Calculate the endpoint of the line
            # yL = y1 - x1 * np.tan(est_angle)
            # yR = y2 + (img_width-x2) * np.tan(est_angle)
            yL = y1 + x1 * np.tan(est_angle)
            yR = y2 - (self.img_width-x2) * np.tan(est_angle)

            y_intercept_final = (yL + yR) / 2
            delta_y = y_intercept_final - newcameramtx[1,2] # Changed to be CY (any offset from the top)
            # psi = np.arctan(delta_y / mtx[1,1]) # horizon pitch
            psi = np.arctan2(delta_y, newcameramtx[1,1]) # horizon pitch
            
            theta = psi + np.pi/2 - np.arcsin(Radius_Earth / (Radius_Earth + self.camera_height)) # pitch
            theta = -theta # Flip sign to match simulator

            roll = est_angle #np.rad2deg(est_angle)
            pitch = theta #np.rad2deg(theta)

            ####################### Debug block: Draw filtered line on the image #######################
            if debug:
                debug_image2 = undistorted.copy()
                debug_image2 = cv2.cvtColor(debug_image2, cv2.COLOR_GRAY2BGR)
                if filtered_lines is not None:
                    for line in filtered_lines:
                        x1, y1, x2, y2 = map(int, line[0])
                        cv2.line(debug_image2, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw in green

                plt.figure()
                plt.scatter([0, self.img_width], [yL, yR], color='red')  # Draw endpoints in red
                plt.scatter([int(self.img_width/2)], [int(y_intercept_final)], color='blue')  # Draw midpoint in blue
                # Display the debug image
                plt.imshow(debug_image2)
                plt.show()
            ####################### End Debug block #######################
        except Exception as e:
            plt.figure()
            plt.imshow(undistorted, cmap='gray')
            plt.title("LSD Failed to Detect Horizon Line")
            # plt.show()
            
            print("LSD Horizon Detection Error:", e)
            plt.figure()
            debug_image = undistorted.copy()
            debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = map(int, line[0])
                    cv2.line(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw in green

            # # Display the debug image
            plt.imshow(debug_image)
            plt.show()
            
            return HorizonObservation(
                valid=False,
                timestamp_sec=timestamp_sec,
                timestamp_nsec=timestamp_nsec,
                roll_rad=0,
                pitch_rad=0,
                roll_var=0.0,
                pitch_var=0.0
            )

        return HorizonObservation(
            valid=True,
            timestamp_sec=timestamp_sec,
            timestamp_nsec=timestamp_nsec,
            roll_rad=roll,
            pitch_rad=pitch,
            # roll_var=np.deg2rad(0.18), #0.02, #0.003
            # pitch_var=np.deg2rad(0.18) # 0.02
            roll_var=np.deg2rad(0.15)**2,
            pitch_var=np.deg2rad(0.15)**2
            # roll_var=np.deg2rad(1), #0.02, #0.003
            # pitch_var=np.deg2rad(1) # 0.02
        )
    #################################

# # Main function for unit testing everything
# if __name__ == "__main__":
#     img_width = 768
#     img_height = 480

#     dist = np.array([0.0, 0.0, 0.0, 0.0])  # Perfect distortion coefficients for simulation
#     FoV = 90  # [deg]
#     f = img_width / (2 * np.tan((FoV * np.pi / 180) / 2))

#     mtx = np.array([
#         [f, 0.0, img_width/2], 
#         [0.0, f, img_height/2], 
#         [0.0, 0.0, 1.0]
#     ])
    
#     # Front image with partial horizon
#     camera_processor_front = ImageProcessor(mtx, dist, img_width, img_height, camera_height=1.57)
#     image_path = 'image_test_r=.69_p=-1.52.png'
#     front_image = cv2.imread(image_path)
#     rpy = [0.69, -1.52, 104.71]  # Pitch, Roll swapped for front camera

#     start_time = time.time()
#     horizon_observation = camera_processor_front.lsd_image_callback(front_image, debug=False)
#     end_time = time.time()
#     print(f"LSD Processing Time: {end_time - start_time:.4f} seconds")
#     print("Expected Processing Time: < 0.1 seconds")

#     print("LSD Roll/Pitch:", np.rad2deg(horizon_observation.roll_rad), np.rad2deg(horizon_observation.pitch_rad))
#     print("Expected Roll/Pitch: 0.69, -1.52")
#     plt.show()

