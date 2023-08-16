#!/usr/bin/env python3
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import serial
import random

# set up the serial connection to the Arduino
ser = serial.Serial('/dev/ttyACM0', 9600)

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  font = cv2.FONT_HERSHEY_SIMPLEX
  fps_avg_frame_count = 10
  
  # Set up camera constants
  IM_WIDTH = 320
  IM_HEIGHT = 240
  detected_left = False
  detected_mid = False
  detected_right = False

  left_counter = 0
  mid_counter = 0
  right_counter = 0

  pause = 0
  pause_counter = 0

  # Initialize the object detection model
  base_options = core.BaseOptions(
      file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
  detection_options = processor.DetectionOptions(
      max_results=3, score_threshold=0.3)
  options = vision.ObjectDetectorOptions(
      base_options=base_options, detection_options=detection_options)
  detector = vision.ObjectDetector.create_from_options(options)

  # Continuously capture images from the camera and run inference
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)

    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)

    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)

    # Draw keypoints and edges on input image
    image = utils.visualize(image, detection_result)
    arr = [0,0,0,0]
    for detection in detection_result.detections:
      # ~ print(detection.categories[0].index)
      obj = detection.categories[0].index
      bbox = detection.bounding_box
      origin_x = bbox.origin_x
      origin_y = bbox.origin_y
      width = bbox.width
      height = bbox.height
      x = int(origin_x+width/2) 
      y = int(origin_y+height/2)
      cv2.circle(image,(x, y), 5, (75,13,180), -1)
      
      # Define left box coordinates (top left and bottom right)
      TL_left = (int(IM_WIDTH*0.016),int(IM_HEIGHT*0.021))
      BR_left = (int(IM_WIDTH*0.323),int(IM_HEIGHT*0.979))

      # Define mid box coordinates (top left and bottom right)
      TL_mid = (int(IM_WIDTH*0.333),int(IM_HEIGHT*0.021))
      BR_mid = (int(IM_WIDTH*0.673),int(IM_HEIGHT*0.979))

      # Define right box coordinates (top left and bottom right)
      TL_right = (int(IM_WIDTH*0.683),int(IM_HEIGHT*0.021))
      BR_right = (int(IM_WIDTH*0.986),int(IM_HEIGHT*0.979))

      cv2.rectangle(image,TL_mid,BR_mid,(255,20,20),3)
      cv2.rectangle(image,TL_left,BR_left,(20,20,255),3)
      cv2.rectangle(image,TL_right,BR_right,(20,255,25),3)
      
      # If object is in inside box, increment left counter variable
      if ((x > TL_left[0]) and (x < BR_left[0]) and (y > TL_left[1]) and (y < BR_left[1])):
            left_counter = left_counter + 1

      # If object is in mid box, increment mid counter variable
      if ((x > TL_mid[0]) and (x < BR_mid[0]) and (y > TL_mid[1]) and (y < BR_mid[1])):
            mid_counter = mid_counter + 1
        
      # If object is in right box, increment right counter variable
      if ((x > TL_right[0]) and (x < BR_right[0]) and (y > TL_right[1]) and (y < BR_right[1])):
            right_counter = right_counter + 1

      if left_counter == 1:
          arr[0]=1    

      if mid_counter == 1:
          arr[1]=1

      if right_counter == 1:
          arr[2]=1

      if obj == 0 or obj == 4:
        if left_counter == 1 or right_counter == 1 or mid_counter == 1:
          arr = [1, 1, 1, 1]
      
      left_counter = 0
      mid_counter = 0
      right_counter = 0

    
    if arr == [0,0,0,0] :
        val = 0
    elif arr == [0,0,1,0] :
        val = 1
    elif arr == [0,1,0,0] :
        val = 2
    elif arr == [1,0,0,0] :
        val = 3
    elif arr == [0,1,1,0] :
        val = 4
    elif arr == [1,0,1,0] :
        val = 5
    elif arr == [1,1,0,0] :
        val = 6
    elif arr == [1,1,1,0] :
        val = 7
    elif arr == [1,1,1,1] :
        val = 8

    val_bytes = str(val).encode('ascii')
    ser.write(val_bytes)

    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
      end_time = time.time()
      fps = fps_avg_frame_count / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Path of the object detection model.',
      required=False,
      default='customedgetpu.tflite')
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, type=int, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      type=int,
      default=320)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=240)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      type=int,
      default=4)
  parser.add_argument(
      '--enableEdgeTPU',
      help='Whether to run the model on EdgeTPU.',
      action='store_true',
      required=False,
      default=True)
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))


if __name__ == '__main__':
  main()
