# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#This is the code block for the class activations.
import cv2
import numpy as np
from ultralytics import YOLO

# class index: class name
  # 0: person
  # 1: bicycle
  # 2: car
  # 3: motorcycle
  # 4: airplane
  # 5: bus
  # 6: train
  # 7: truck
  # 8: boat
  # 9: traffic light
  # 10: fire hydrant
  # 11: stop sign
  # 12: parking meter
  # 13: bench
  # 14: bird
  # 15: cat
  # 16: dog
  # 17: horse
  # 18: sheep
  # 19: cow
  # 20: elephant
  # 21: bear
  # 22: zebra
  # 23: giraffe
  # 24: backpack
  # 25: umbrella
  # 26: handbag
  # 27: tie
  # 28: suitcase
  # 29: frisbee
  # 30: skis
  # 31: snowboard
  # 32: sports ball
  # 33: kite
  # 34: baseball bat
  # 35: baseball glove
  # 36: skateboard
  # 37: surfboard
  # 38: tennis racket
  # 39: bottle
  # 40: wine glass
  # 41: cup
  # 42: fork
  # 43: knife
  # 44: spoon
  # 45: bowl
  # 46: banana
  # 47: apple
  # 48: sandwich
  # 49: orange
  # 50: broccoli
  # 51: carrot
  # 52: hot dog
  # 53: pizza
  # 54: donut
  # 55: cake
  # 56: chair
  # 57: couch
  # 58: potted plant
  # 59: bed
  # 60: dining table
  # 61: toilet
  # 62: tv
  # 63: laptop
  # 64: mouse
  # 65: remote
  # 66: keyboard
  # 67: cell phone
  # 68: microwave
  # 69: oven
  # 70: toaster
  # 71: sink
  # 72: refrigerator
  # 73: book
  # 74: clock
  # 75: vase
  # 76: scissors
  # 77: teddy bear
  # 78: hair drier
  # 79: toothbrush

def extract_people(input_video_path, output_path, verbose=False):
  target_Classes = ['person']
  extract_objects(input_video_path, output_path, target_Classes, verbose)

def extract_vehicles(input_video_path, output_path, verbose=False):
  target_Classes = ['bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
  extract_objects(input_video_path, output_path, target_Classes, verbose)

def extract_animals(input_video_path, output_path, verbose=False):
  target_Classes = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']
  extract_objects(input_video_path, output_path, target_Classes, verbose)

def extract_handheld(input_video_path, output_path, verbose=False):
  target_Classes = [
        # Accessories & Toys
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        # Sports (Handheld)
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'tennis racket',
        # Food & Kitchen
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake',
        # Small Electronics
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        # Misc Personal
        'book', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
  extract_objects(input_video_path, output_path, target_Classes, verbose)

def extract_nonhandheld(input_video_path, output_path, verbose=False):
  target_Classes = [
        # Street Infrastructure
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        # Rideable Boards (Feet)
        'skis', 'snowboard', 'skateboard', 'surfboard',
        # Furniture
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        # Appliances & Decor
        'tv', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'clock', 'vase'
    ]
  extract_objects(input_video_path, output_path, target_Classes, verbose)

def extract_objects(input_video_path, output_path, target_classes, verbose=False):
  s_mode = 0   # 0: semantic, 1: instance

  cap = cv2.VideoCapture(input_video_path)

  model = YOLO("yolo11l-seg.pt")
  names = model.names
  target_Indices = []
  target_Classes = target_classes

  for Name in target_Classes:
    try:
      index = list(names.values()).index(Name)
      target_Indices.append(index)
    except ValueError:
      print(f"'{Name}' is not found.")

  # Video writer
  w, h, fps = (int(cap.get(x)) for x in
              (cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS))
  vw = cv2.VideoWriter(output_path,
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps,
                      (w, h))

  while cap.isOpened():

      success, im0 = cap.read()
      if not success:
          break


      mask_class_im0 = np.zeros((h,w), dtype=np.uint8) # can be removed for debugger (seeing image as well)
      results = model.track(im0, persist=True, imgsz=1280, classes=target_Indices, conf=0.25, verbose=False)[0]

      # Only draw mask+bbox if track id exists
      if results.boxes.id is not None:

          # Extract box, trackid, class and masks
    #     boxes = results.boxes.xyxy.tolist() #enable this and ''' 'ed code for the bounding boxes+names to be visible.
          masks = results.masks

          if results.boxes.id is not None:
            masks = results.masks
            if masks is not None:
              for mask_tensor in masks.data:
                  mask_sizer = mask_tensor.cpu().numpy().astype(np.uint8) * 255
                  mask_np = cv2.resize(mask_sizer, (w, h), interpolation=cv2.INTER_NEAREST)
                  mask_class_im0 = np.maximum(mask_class_im0, mask_np)

      #cv2_imshow(mask_class_im0) #should be cv2.imshow("Image segementation", mask_class_im0) in actual code, or same with mask_class_im0 for machine input
      final_output = cv2.cvtColor(mask_class_im0, cv2.COLOR_GRAY2BGR)
      vw.write(final_output)
      if cv2.waitKey(1) & 0xFF == ord("q"):
          break

  cap.release()
  cv2.destroyAllWindows()
