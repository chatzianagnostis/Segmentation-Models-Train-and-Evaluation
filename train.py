import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from dataset import create_dataset
from utils import load_config
from architecture import CustomSegmentationModel


def main(config_file):
    # Read from config.yaml====================================================================
    config_project = load_config(args.config_project)
    config_model= load_config(args.config_model)

    # Directory paths:
    DIR = config_project["DIR"]
    DATASET_PATH = config_project["DATASET_PATH"]
    TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
    VALID_PATH = os.path.join(DATASET_PATH, 'valid')
    IMAGE_EXTENSION = config_project["IMAGE_EXTENSION"]
    MAKS_EXTENSION = config_project["MAKS_EXTENSION"]
    EXTENSIONS = [IMAGE_EXTENSION, MAKS_EXTENSION]
    # Model hyperparameters:
    MULTICLASS_MODE = config_project["MODEL"]["MULTICLASS_MODE"]
    EXP_NAME = config_project["MODEL"]["EXP_NAME"]
    CLASSES = config_project["MODEL"]["CLASSES"]
    BATCH_SIZE = config_project["MODEL"]["BATCH_SIZE"]
    LEARNING_RATE = config_project["MODEL"]["LEARNING_RATE"]
    EPOCHS = config_project["MODEL"]["EPOCHS"]

    MODEL = config_model['ARCHITECTURE']
    KWARGS = config_model['HYP']

    # Create folder for train resuts =========================================================
    OUTPUT_FOLDER = os.path.join(DIR, 'runs','train', EXP_NAME)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    SAVE_MODEL_PATH = os.path.join(OUTPUT_FOLDER, 'model')
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)

    # Define tranforms using Albumations =====================================================
    test_transform = A.Compose([
        A.Resize(1376, 800)
    ])

    train_transform = A.Compose(
        [
            A.Resize(1376, 800),
            #A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5), #TODO
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            #A.MedianBlur(blur_limit=3, always_apply=False, p=0.1),
        ]
    )

    # Create datasets and define dataloaders =================================================
    train_dataset = create_dataset(
        dataset_path=TRAIN_PATH,
        transform = train_transform,
        classes = CLASSES,
        extensions = EXTENSIONS
    )

    valid_dataset = create_dataset(
        dataset_path=VALID_PATH,
        transform = test_transform,
        classes = CLASSES,
        extensions = EXTENSIONS
    )

    train_set = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

    valid_set = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

    # Initiate UNet++ Model ==================================================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_properties(0))

    segmentation_model = CustomSegmentationModel(MODEL, **KWARGS)
    model = segmentation_model.build_model()
    model.to(DEVICE)

    torch.save(model, f'{SAVE_MODEL_PATH}/init_model.pth')

    #summary(model, input_size=(3, 1376, 800), device=DEVICE.type)
    print(f'Dataset stats:\n Training Set: {len(train_dataset)} images\n Validation Set: {len(valid_dataset)} images')

    # Define Loss and Metrics to Monitor =====================================================
    loss = smp.losses.TverskyLoss(mode=MULTICLASS_MODE)
    loss.__name__ = 'TverskyLoss'

    metrics=[] #TODO

    # Define Optimizer =======================================================================
    optimizer = optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    # Define epochs ==========================================================================
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss,
        metrics= metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # Train model ============================================================================
    best_loss = float('inf')
    writer = SummaryWriter(OUTPUT_FOLDER)
    print('Starting Training ...')

    for i in range(0, EPOCHS):
        print(f'Epoch:{i+1}/{EPOCHS}')
        train_logs = train_epoch.run(train_set)
        valid_logs = valid_epoch.run(valid_set)
        writer.add_scalar('TverskyLoss/train', train_logs['TverskyLoss'], i)
        writer.add_scalar('TverskyLoss/val', valid_logs['TverskyLoss'], i)

        # Save the model after each epoch
        torch.save(model, f'{SAVE_MODEL_PATH}/last_model.pth')

        # Save the model with the best loss
        if valid_logs['TverskyLoss'] < best_loss:
            best_loss = valid_logs['TverskyLoss']
            torch.save(model, f'{SAVE_MODEL_PATH}/best_model.pth')

    writer.close()

    print(f'TverskyLoss loss from best model in validation set: {best_loss}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unet++ with custom dataset")
    parser.add_argument("--config", dest="config_project", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--model", dest="config_model", default="models/unetplusplus.hyp.yaml", help="Path to YAML config model file")

    args = parser.parse_args()
    main(args)
