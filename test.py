import argparse
import os
import numpy as np

import torch
import torch.nn as nn
from torchsummary import summary

import albumentations as A
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import ssl

from utils import load_config, seg2bbox, seg2bbox_with_conf, calculate_iou, calculate_metrics_per_class, visualize_segments
from dataset import create_dataset

def main(config_file):
    # Read from config.yaml===================================================================
    config = load_config(config_file)
    # Directory paths:
    DIR = config["DIR"]
    DATASET_PATH = config["DATASET_PATH"]
    TEST_PATH = os.path.join(DATASET_PATH, 'test')
    IMAGE_EXTENSION = config["IMAGE_EXTENSION"]
    MAKS_EXTENSION = config["MAKS_EXTENSION"]
    EXTENSIONS = [IMAGE_EXTENSION, MAKS_EXTENSION]

    # Model hyperparameters:
    MULTICLASS_MODE = config["MODEL"]["MULTICLASS_MODE"]
    EXP_NAME = config["MODEL"]["EXP_NAME"]
    CLASSES = config["MODEL"]["CLASSES"]
    MODEL_PATH = os.path.join(DIR,'runs','train', EXP_NAME, 'model', 'best_model.pth')

    # Create folder for train resuts =========================================================
    OUTPUT_FOLDER = os.path.join(DIR, 'runs','test', EXP_NAME)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    DETECT_FOLDER = os.path.join(OUTPUT_FOLDER,'detect')
    if not os.path.exists(DETECT_FOLDER):
        os.makedirs(DETECT_FOLDER)

    # Define tranforms using Albumations =====================================================
    test_transform = A.Compose([
        A.Resize(1376, 800)
    ])

    # Create datasets and define dataloaders =================================================
    test_dataset = create_dataset(
        dataset_path=TEST_PATH,
        transform = test_transform,
        classes = CLASSES,
        extensions = EXTENSIONS
    )

    test_set = torch.utils.data.DataLoader(test_dataset, batch_size= 1, shuffle=True, sampler=None,
                batch_sampler=None, num_workers=0, collate_fn=None,
                pin_memory=False, drop_last=False, timeout=0,
                worker_init_fn=None)
        
    # Load model =============================================================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(MODEL_PATH, map_location=torch.device('cuda'))
    summary(model, input_size=(3, 1376, 800), device=DEVICE.type)
    print(f"Loading moodel : {MODEL_PATH}")



    # Define Loss and Metrics to Monitor =====================================================
    loss = smp.losses.TverskyLoss(mode=MULTICLASS_MODE)
    loss.__name__ = 'TverskyLoss'

    metrics=[] #TODO

    # Test Epoch =============================================================================
    test_epoch = smp.utils.train.ValidEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    )
    print('Testing ...')
    logs = test_epoch.run(test_set)
    print(logs)

    precisions_per_classes = []
    recalls_per_class = []

    class_dict = {i: CLASSES[i] for i in range(len(CLASSES))}
    import pdb
    for i in range(len(test_dataset)):
        image_vis = test_dataset[i][0].permute(1,2,0)

        image_vis = image_vis.numpy()*255
        image_vis = image_vis.astype('uint8')
        image, gt_mask = test_dataset[i]
    
        gt_mask = (gt_mask.squeeze().cpu().numpy().round())
        gt_bboxes, gt_classes = seg2bbox(gt_mask)
        x_tensor = image.to(DEVICE).unsqueeze(0)
        pr_mask = model.predict(x_tensor)
        m = nn.Softmax(dim=1)
        pr_probs = m(pr_mask)
        pr_mask = torch.argmax(pr_probs, dim=1).squeeze(1)
        pr_mask = (pr_mask.squeeze().cpu().numpy().round())
        pr_bboxes, pr_classes, confidences = seg2bbox_with_conf(pr_probs)
        precision_per_classes, recall_per_class = calculate_metrics_per_class(gt_bboxes, gt_classes, pr_bboxes, pr_classes, class_dict, len(CLASSES))
        precisions_per_classes.append(precision_per_classes)
        recalls_per_class.append(recall_per_class)
        # save_path = f'{DETECT_FOLDER}/image_{i}.png'
        # visualize_segments(image_vis, pr_mask, class_dict,save_path)

    average_map50_per_class = {class_name: np.mean([d.get(class_name) for d in precisions_per_classes if d.get(class_name) is not None]) for class_name in class_dict.values()}
    average_recall_per_class = {class_name: np.mean([d.get(class_name) for d in recalls_per_class if d.get(class_name) is not None]) for class_name in class_dict.values()}
    for key, value in average_map50_per_class.items():
        print(f'{key}: Precision {value:.3f} and Recall {average_recall_per_class[key]:.3f}')
    
    # average_map50_values = list(average_map50_per_class.values())[1:]
    # print(f'mAP@50 All: {np.mean(average_map50_values):.3f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Unet++ with custom dataset")
    parser.add_argument("--config", dest="config_file", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config_file)
