import sys
sys.path.append('D:\cocoapi\PythonAPI')
from pycocotools.coco import COCO

# initialize COCO API for instance annotations
instances_annFile = r'C:\Users\vJanGo\Desktop\python_big_work\Image-Captioning-main\cocoapi\annotations\instances_val2014.json'
coco = COCO(instances_annFile)
# initialize COCO API for caption annotations
captions_annFile = r'C:\Users\vJanGo\Desktop\python_big_work\Image-Captioning-main\cocoapi\annotations\captions_val2014.json'
coco_caps = COCO(captions_annFile)

# get image ids 
ids = list(coco.anns.keys())