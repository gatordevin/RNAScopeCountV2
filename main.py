import MainApp
from pathlib import Path
import datetime
import json
from PIL import Image
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper
import detectron2.data.transforms as T
import copy
from detectron2.data import detection_utils as utils
import torch
import random
import math
import numpy as np
from detectron2.engine import DefaultPredictor

cfg = get_cfg()

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def add_polygons_to_dataset(polygon_list):
    now = datetime.datetime.now()
    dataset_folder = Path("datasets/")
    dataset_file = dataset_folder / "default.json"

    dataset_folder.mkdir(parents=True, exist_ok=True)

    coco_dataset_dict = {}

    if not dataset_file.exists():
        print("Creating dataset")

        coco_dataset_dict["info"] = {}
        coco_dataset_dict["info"]["description"] = "default dataset file"
        coco_dataset_dict["info"]["url"] = ""
        coco_dataset_dict["info"]["version"] = "1.0"
        coco_dataset_dict["info"]["year"] = str(now.year)
        coco_dataset_dict["info"]["contributor"] = ""
        coco_dataset_dict["info"]["data_created"] = str(now.month) + "/" + str(now.day)

        coco_dataset_dict["licenses"] = []
        coco_dataset_dict["images"] = []
        coco_dataset_dict["annotations"] = []
        coco_dataset_dict["categories"] = []

        with open(str(dataset_file), 'w') as outfile:
            json.dump(coco_dataset_dict, outfile)
    
    with open(str(dataset_file)) as f:
        coco_dataset_dict = json.load(f)

    polygon_dict = {}
    for polygon in polygon_list:
        if not polygon.image in polygon_dict:
            polygon_dict[polygon.image] = []
        polygon_dict[polygon.image].append([polygon.category, polygon.points])
    
    #Add Data to COCO dataset
    existing_image_names = [image["file_name"] for image in coco_dataset_dict["images"]]
    existing_category_names = [image["name"] for image in coco_dataset_dict["categories"]]
    for image_name in polygon_dict:
        image = Image.open(image_name)
        if not image_name in existing_image_names:

            image_dict = {}
            image_dict["file_name"] = image_name
            image_dict["height"] = image.size[0]
            image_dict["width"] = image.size[1]
            image_dict["id"] = len(coco_dataset_dict["images"])
            coco_dataset_dict["images"].append(image_dict)
            existing_image_names.append(image_name)
        
        for polygon in polygon_dict[image_name]:
            if not polygon[0] in existing_category_names:
                category_dict = {}
                category_dict["id"] = len(coco_dataset_dict["categories"])
                category_dict["name"] = polygon[0]
                coco_dataset_dict["categories"].append(category_dict)
                existing_category_names.append(polygon[0])
            
            annotation_dict = {}
            annotation_dict["segmentation"] = []
            
            scaled_points = [[point[0]*image.size[1],point[1]*image.size[0]] for point in polygon[1]] 
            coco_points = []
            for point in scaled_points:
                coco_points.extend(point)

            annotation_dict["segmentation"].append(coco_points)
            annotation_dict["area"] = PolygonArea(scaled_points)
            annotation_dict["iscrowd"] = 0
            annotation_dict["image_id"] = existing_image_names.index(image_name)

            x_coordinates, y_coordinates = zip(*scaled_points)
            annotation_dict["bbox"] = [min(x_coordinates), min(y_coordinates), max(x_coordinates)-min(x_coordinates), max(y_coordinates)-min(y_coordinates)]
            annotation_dict["category_id"] = existing_category_names.index(polygon[0])
            if len(coco_dataset_dict["annotations"]) > 0:
                annotation_dict["id"] = coco_dataset_dict["annotations"][-1]["id"]+1
            else:
                annotation_dict["id"] = 0
            coco_dataset_dict["annotations"].append(annotation_dict)
        
  
    #Add Categories to COCO dataset

    with open(str(dataset_file), 'w') as outfile:
        json.dump(coco_dataset_dict, outfile)
    
def remove_polygon_from_dataset(polygon):
    dataset_folder = Path("datasets/")
    dataset_file = dataset_folder / "default.json"
    coco_dataset_dict = {}

    with open(str(dataset_file)) as f:
        coco_dataset_dict = json.load(f)

    y, x = Image.open(polygon.image).size
    scaled_points = [[point[0]*x,point[1]*y] for point in polygon.points]
    coco_points = []
    for point in scaled_points:
        coco_points.extend(point)

    for annotation in coco_dataset_dict["annotations"]:
        for polygon in annotation["segmentation"]:
            if(polygon==coco_points):
                coco_dataset_dict["annotations"].remove(annotation)

    with open(str(dataset_file), 'w') as outfile:
        json.dump(coco_dataset_dict, outfile)

# def get_cropped_images(img, x ,y):
#     y_size, x_size, col = img.shape

#     overlap = 40
#     x_num = x_size//x
#     total_x_diff = x_size - (x_num * x)
#     target_diff = (x_num-1)*overlap
#     target_vs_total = target_diff-total_x_diff
#     if(target_vs_total>0):
#         x_num += math.ceil(target_vs_total/x)

#     y_num = y_size//y
#     total_y_diff = y_size - (y_num * y)
#     target_diff = (y_num-1)*overlap
#     target_vs_total = target_diff-total_y_diff
#     if(target_vs_total>0):
#         y_num += math.ceil(target_vs_total/y)

#     x_increment = int((x_size-x)/(x_num-1))
#     y_increment = int((y_size-y)/(y_num-1))
#     for x_crop in range(x_num):
#         start_x = x_crop*x_increment
#         for y_crop in range(y_num):
#             start_y = y_crop*y_increment
#             cropped_image = img[start_y:start_y+y,start_x:start_x+x]
#             yield [cropped_image, x_crop, y_crop]

# def custom_mapper(dataset_dict):
#     print(dataset_dict)
#     dataset_dict = copy.deepcopy(dataset_dict)
#     image = utils.read_image(dataset_dict["file_name"], format="BGR")
#     for cropped_image, x_crop, y_crop in get_cropped_images(image, random.randrange(300, 600), random.randrange(300, 600)):
#         print(cropped_image.shape)
#         dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
#         instances = utils.annotations_to_instances(dataset_dict.pop("annotations"), image.shape[:2])
#         dataset_dict["instances"] = utils.filter_empty_instances(instances)
#     return dataset_dict

class Trainer(DefaultTrainer):

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[T.RandomCrop("absolute_range", (300, 600)),T.RandomRotation(random.randrange(0, 360)), T.RandomContrast(0.5, 1.5), T.RandomSaturation(0.5, 1.5)]))

def train(polygons):
    dataset_folder = Path("datasets/")
    dataset_file = dataset_folder / "default.json"
    register_coco_instances(dataset_file.name, {}, dataset_file, "")

    cfg.DATASETS.TRAIN = (dataset_file.name,)
    cfg.DATASETS.TEST = ()
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    DatasetCatalog.remove(dataset_file.name)

def run(image_name, add_polygon):
    predictor = DefaultPredictor(cfg)
    img = Image.open(image_name)
    y, x = img.size
    img = img.convert("RGB")
    img = np.asarray(img)
    img = img[:, :, ::-1]
    outputs = predictor(img)
    for box in outputs["instances"].to("cpu").get_fields()["pred_boxes"]:
        x1, y1, x2, y2 = box.numpy()
        polygon = [[x1/x,y1/y],[x2/x,y1/y],[x2/x,y2/y],[x1/x,y2/y]]
        add_polygon(polygon, image_name)

if __name__ == '__main__':
    cfg.merge_from_file("/home/techgarage/Projects/Max Planck/RNAScopeCountV2/models/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATALOADER.NUM_WORKERS = 2
    default_weight_path = Path("models/model_final.pth")
    if(default_weight_path.exists()):
        cfg.MODEL.WEIGHTS = str(default_weight_path)
    else:
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.005
    cfg.SOLVER.MAX_ITER = 1000
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (64)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = str(default_weight_path.parent)

    dataset_folder = Path("datasets/")
    dataset_file = dataset_folder / "default.json"
    MainApp.MyApp(add_polygons_to_dataset, remove_polygon_from_dataset, train, dataset_file, run).run()

