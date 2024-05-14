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


def main(config_file):
    # Read from config.yaml====================================================================
    config = load_config(config_file)
    # Directory paths:
    DIR = config["DIR"]
    DATASET_PATH = config["DATASET_PATH"]
    TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
    VALID_PATH = os.path.join(DATASET_PATH, 'valid')
    IMAGE_EXTENSION = config["IMAGE_EXTENSION"]
    MAKS_EXTENSION = config["MAKS_EXTENSION"]
    EXTENSIONS = [IMAGE_EXTENSION, MAKS_EXTENSION]
    # Model hyperparameters:
    MULTICLASS_MODE = config["MODEL"]["MULTICLASS_MODE"]
    EXP_NAME = config["MODEL"]["EXP_NAME"]
    ENCODER = config["MODEL"]["ENCODER"]
    ENCODER_WEIGHTS = config["MODEL"]["ENCODER_WEIGHTS"]
    CLASSES = config["MODEL"]["CLASSES"]
    ACTIVATION = config["MODEL"]["ACTIVATION"]
    BATCH_SIZE = config["MODEL"]["BATCH_SIZE"]
    LEARNING_RATE = config["MODEL"]["LEARNING_RATE"]
    EPOCHS = config["MODEL"]["EPOCHS"]
    CHANNELS = config["MODEL"]["CHANNELS"]

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

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS, 
        in_channels=CHANNELS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    ).to(DEVICE)

    torch.save(model, f'{SAVE_MODEL_PATH}/init_model.pth')

    summary(model, input_size=(3, 1376, 800), device=DEVICE.type)
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
    parser.add_argument("--config", dest="config_file", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config_file)