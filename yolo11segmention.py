import cv2
from PIL import Image

from ultralytics import YOLO
import pyresearch 

model = YOLO("last.pt")


results = model.predict(source="demo.mp4", show=True,save=True,    hide_conf=False)  # Display preds. Accepts all YOLO predict arguments

