import yaml
import numpy as np
import torch
from skimage.measure import label, regionprops
from sklearn.metrics import average_precision_score
from skimage.measure import find_contours
import matplotlib.pyplot as plt

def load_config(config_file):
    """
    Load configuration settings from a YAML file.
    
    Parameters:
    config_file (str): Path to the YAML configuration file.
    
    Returns:
    dict: Configuration settings.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


import pdb
def seg2bbox(segm):
    """
    Convert segmentation masks to bounding boxes and corresponding classes.
    
    Parameters:
    segm (numpy.ndarray): Segmentation mask.
    
    Returns:
    tuple: Tuple containing lists of bounding boxes and corresponding classes.
    """
    labels = label(segm)
    props = regionprops(labels)
    bboxes = []
    classes = []
    
    for prop in props:
        x1, y1, x2, y2 = prop.bbox
        bboxes.append([x1, y1, x2, y2])
        class_in_bbox = np.argmax(np.bincount(segm[x1:x2, y1:y2].flatten()))
        if class_in_bbox == 0:
            class_in_bbox = 7
        classes.append(class_in_bbox)
    return bboxes, classes


def seg2bbox_with_conf(pr_probs):
    """
    Convert segmentation masks to bounding boxes and corresponding classes.
    
    Parameters:
    pr_probs (torch.Tensor): Prediction probabilities tensor of shape (1, num_classes, height, width).
    
    Returns:
    tuple: Tuple containing lists of bounding boxes and corresponding classes and confidences.
    """
    # Convert prediction probabilities tensor to numpy array
    pr_probs_np = pr_probs.squeeze(0).cpu().numpy()

    bboxes = []
    classes = []
    confidences = []

    # Iterate over each class (excluding background class)
    for class_index in range(1, pr_probs_np.shape[0]):
        # Get segmentation mask for the current class
        segm = (np.argmax(pr_probs_np, axis=0) == class_index).astype(np.uint8)
        # Label connected components in the segmentation mask
        labels = label(segm)
        props = regionprops(labels)
        
        # Iterate over each bounding box region
        for prop in props:
            x1, y1, x2, y2 = prop.bbox
            # Compute confidence as the median probability within the bounding box region
            confidence = np.median(pr_probs_np[class_index, x1:x2, y1:y2])
            bboxes.append([x1, y1, x2, y2])
            classes.append(class_index)
            confidences.append(confidence)

    return bboxes, classes, confidences


def calculate_iou(bbox1, bbox2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.
    
    Parameters:
    bbox1 (list): Coordinates of the first bounding box in the format [min_row, min_col, max_row, max_col].
    bbox2 (list): Coordinates of the second bounding box in the format [min_row, min_col, max_row, max_col].
    
    Returns:
    float: Intersection over Union (IoU) score.
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    iou = intersection / float(area_bbox1 + area_bbox2 - intersection)
    return iou


def calculate_metrics_per_class(gt_bboxes, gt_classes, pr_bboxes, pr_classes, class_dict, num_classes, iou_thres=0.1):
    """
    Calculate class-wise Precision and Recall for each class.

    Parameters:
    gt_bboxes (list): Ground truth bounding boxes.
    gt_classes (list): Ground truth class labels.
    pr_bboxes (list): Predicted bounding boxes.
    pr_classes (list): Predicted class labels.
    class_dict (dict): A dictionary mapping class indices to class names.
    num_classes (int): Number of classes.
    iou_thres (float): IoU threshold for considering a detection as correct.

    Returns:
    dict: A dictionary containing class-wise Precision and Recall scores.
    """
    precisions = {}
    recalls = {}
    
    for c in range(num_classes):
        y_true = [1 if x == c else 0 for x in gt_classes]
        y_score = []
        true_positives = 0
        false_negatives = 0
        false_positives = 0
        
        for i in range(len(gt_classes)):
            if gt_classes[i] == c:
                max_iou = 0
                max_j = -1
                for j in range(len(pr_classes)):
                    if pr_classes[j] == c:
                        iou = calculate_iou(gt_bboxes[i], pr_bboxes[j])
                        if iou > max_iou:
                            max_iou = iou
                            max_j = j
                if max_j != -1:
                    if max_iou > iou_thres:
                        true_positives += 1
                        y_score.append(max_iou)
                        pr_classes[max_j] = -1
                    else:
                        false_negatives += 1
                        y_score.append(0)
                else:
                    false_negatives += 1
                    y_score.append(0)
            else:
                false_positives += 1
                y_score.append(0)

        if max(y_true) > 0:
            precision = average_precision_score(y_true, y_score)
            precisions[class_dict[c]] = precision
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            recalls[class_dict[c]] = recall

    return precisions, recalls
    

def visualize_segments(image, output, class_dict, save_path):
    """
    Visualize the output of a U-Net++ model.
    
    Parameters:
    image (numpy.ndarray): The original image.
    output (numpy.ndarray): The output of the U-Net++ model. Assumes a 2D array where each unique value represents a different class.
    class_dict (dict): A dictionary mapping class labels to class names.
    save_path (str): Path to save the visualization (optional).
    """
    # Ensure output is a numpy array
    if not isinstance(output, np.ndarray):
        output = np.array(output)
    
    # Define the colormap - this can be adjusted as needed
    cmap = plt.get_cmap('tab10')
    
    # Display the original image
    plt.imshow(image, cmap='gray')
    
    # Find contours for each segment
    for class_label in np.unique(output):
        contours = find_contours(output == class_label, 0.5)
        
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=cmap(class_label))
            
    # Create a legend showing the class names
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap(class_label)) for class_label in class_dict.keys()]
    plt.legend(handles, class_dict.values(), bbox_to_anchor=(1.05, 1), loc='upper left')

    #plt.show()
    plt.savefig(save_path)
    plt.close()
