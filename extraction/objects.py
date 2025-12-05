# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
#This is the code block for the class activations.
import cv2
import numpy as np
from ultralytics import YOLO

def extract_objects(input_video_path, output_path):
  s_mode = 0   # 0: semantic, 1: instance

  cap = cv2.VideoCapture(input_video_path)

  model = YOLO("yolo11n-seg.pt")
  names = model.names
  target_Indices = []
  target_Classes = ['person'] #for this code's example, try just person.

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
      results = model.track(im0, persist=True)[0]

      # Only draw mask+bbox if track id exists
      if results.boxes.id is not None:

          # Extract box, trackid, class and masks
    #     boxes = results.boxes.xyxy.tolist() #enable this and ''' 'ed code for the bounding boxes+names to be visible.
          tids = results.boxes.id.int().tolist()
          clss = results.boxes.cls.cpu().tolist()
          masks = results.masks

          if masks is not None:
            for class_index, mask_tensor in zip(clss, masks.data): #iterate through all the found objects
              if int(class_index) in target_Indices:
                mask_sizer = mask_tensor.cpu().numpy().astype(np.uint8) * 255 #used to prevent re-size errors
                mask_np = cv2.resize(mask_sizer, (w, h), interpolation=cv2.INTER_NEAREST) #get a numpy array up, and scale.
                mask_class_im0 = np.maximum(mask_class_im0, mask_np) #combine the masks.

            '''
          for b, t, c in zip(boxes, tids, clss):
              annotator.box_label(
                  b,
                  color=colors(t if s_mode==1 else c, True),
                  label=names[c])
  '''

      #cv2_imshow(mask_class_im0) #should be cv2.imshow("Image segementation", mask_class_im0) in actual code, or same with mask_class_im0 for machine input
      final_output = cv2.cvtColor(mask_class_im0, cv2.COLOR_GRAY2BGR)
      vw.write(final_output)
      if cv2.waitKey(1) & 0xFF == ord("q"):
          break

  cap.release()
  cv2.destroyAllWindows()
