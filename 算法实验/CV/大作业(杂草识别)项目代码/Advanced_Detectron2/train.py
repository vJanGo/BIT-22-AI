# 训练模型

import argparse
import torch, detectron2
import os

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from dataset_init import get_weed_dataset_dicts
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

# 参数
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Detectron2 model")
    parser.add_argument("--train_images", type=str, default='train/images', help="Path to training images")
    parser.add_argument("--train_labels", type=str, default='train/labels', help="Path to training labels")
    parser.add_argument("--test_images", type=str, default='test/images', help="Path to test images")
    parser.add_argument("--test_labels", type=str, default='test/labels', help="Path to test labels")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--ims_per_batch", type=int, default=16, help="Images per batch")
    parser.add_argument("--base_lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--epochs", type=int, default=6000, help="Maximum number of iterations")
    parser.add_argument("--batch_size_per_image", type=int, default=16, help="Batch size per image")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use")
    return parser.parse_args()

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

def main():
    args = parse_args()
    for d in ['train']:
        # 加载训练集
        DatasetCatalog.register("weed_" + d, lambda d=d: get_weed_dataset_dicts(args.train_images, args.train_labels))
        MetadataCatalog.get("weed_" + d).set(thing_classes=["weed", "mq"])

    # 训练准备
    cfg = get_cfg() 
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("weed_train",)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = args.ims_per_batch
    cfg.SOLVER.BASE_LR = args.base_lr
    cfg.SOLVER.MAX_ITER = args.epochs
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.batch_size_per_image
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes
    cfg.MODEL.DEVICE = args.device
    # 开始训练
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    main()

