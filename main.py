import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from dataset import MEGDataset
from sklearn.model_selection import StratifiedKFold
import random
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from utils import AverageMeter, timeSince, get_logger
import time
import gc
from models import CustomModel
from torch.optim.lr_scheduler import OneCycleLR

class CFG:
    AMP = False
    PRINT_FREQ = 20
    model_name = "resnet34d"
    img_size = 512
    EPOCHS = 9
    OUTPUT_DIR = 'output/'
    batch_size = 32
    lr = 1.0e-03
    weight_decay = 1.0e-02
    es_patience =  5
    seed = 1086
    deterministic = True
    enable_amp = True
    device = "cuda"
    VISUALIZE = False
    
def set_random_seed(seed: int = 42, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    
dataset_dir = 'B:/datasets/biofind/'
Classses = ['HEC', 'MCI']
num_classes = len(Classses)

files = os.listdir(dataset_dir) 
print(f'There are {len(files)} ')
LOGGER = get_logger()
    
if CFG.VISUALIZE:
    for (raw_data, spectrograms, mri, label) in train_dataloader:
        plt.figure(figsize=(20,8))
        
        plt.subplot(1, 3, 1)
        plt.imshow(raw_data[0,:])
        
        plt.subplot(1, 3, 2)
        plt.imshow(spectrograms[:,:,0])
        
        plt.subplot(1, 3, 3)
        plt.imshow(mri[:,:,60])        
        
        plt.xlabel('Time (sec)',size=16)
        plt.show()
        
        print('label:', label)
        break 

def train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    """One epoch training pass."""
    model.train() 
    criterion = nn.KLDivLoss(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.AMP)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    
    # ========== ITERATE OVER TRAIN BATCHES ============
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, (raw_data, spectrograms, mri, y_true) in enumerate(tqdm_train_loader):
            raw_data = raw_data.to(device)
            spectrograms = spectrograms.to(device)
            mri = mri.to(device)
            y_true = y_true.to(device)
            batch_size = label.size(0)
            with torch.cuda.amp.autocast(enabled=CFG.AMP):
                y_preds = model(raw_data, spectrograms, mri) 
                loss = criterion(F.log_softmax(y_preds, dim=1), y_true)
            losses.update(loss.item(), batch_size)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            scheduler.step()
            end = time.time()

            # ========== LOG INFO ==========
            if step % CFG.PRINT_FREQ == 0 or step == (len(train_loader)-1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      'LR: {lr:.8f}  '
                      .format(epoch+1, step, len(train_loader), 
                              remain=timeSince(start, float(step+1)/len(train_loader)),
                              loss=losses,
                              lr=scheduler.get_last_lr()[0]))

    return losses.avg


def valid_epoch(valid_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    start = time.time()
    with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
        for step, (raw_data, spectrograms, mri, y_true) in enumerate(tqdm_valid_loader):
            raw_data = raw_data.to(device)
            spectrograms = spectrograms.to(device)
            mri = mri.to(device)
            y_true = y_true.to(device)
            batch_size = label.size(0)
            with torch.cuda.amp.autocast(enabled=CFG.AMP):
                y_preds = model(raw_data, spectrograms, mri) 
                loss = criterion(F.log_softmax(y_preds, dim=1), y_true)
            losses.update(loss.item(), batch_size)

            # ========== LOG INFO ==========
            if step % CFG.PRINT_FREQ == 0 or step == (len(valid_loader)-1):
                print('EVAL: [{0}/{1}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.avg:.4f} '
                      .format(step, len(valid_loader),
                              remain=timeSince(start, float(step+1)/len(valid_loader)),
                              loss=losses))
    return losses.avg  
    
def train(CFG):
    skf = StratifiedKFold(n_splits=5)
    for i, (train_idx, val_idx) in enumerate(skf.split(files)):
        trainset, valset = files[train_idx],files[val_idx]
        train_dataset = MEGDataset(trainset, mode='train', Normalize=True)
        val_dataset = MEGDataset(valset, mode='val', Normalize=True)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

        print(f'Trainset: {len(trainset)} Valset: {len(valset)}')
        # ======== MODEL ==========
        model = CustomModel(CFG)
        model.to(CFG.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=config.WEIGHT_DECAY)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=1e-3,
            epochs=CFG.EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
            final_div_factor=100,
        )
        # ======= LOSS ==========
        criterion = nn.KLDivLoss(reduction="batchmean")
        best_loss = np.inf
        # ====== ITERATE EPOCHS ========
        for epoch in range(config.EPOCHS):
            start_time = time.time()

            # ======= TRAIN ==========
            avg_train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, scheduler, device)

            # ======= EVALUATION ==========
            avg_val_loss, prediction_dict = valid_epoch(valid_loader, model, criterion, device)
            predictions = prediction_dict["predictions"]
            
            # ======= SCORING ==========
            elapsed = time.time() - start_time

            LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
                torch.save({'model': model.state_dict(),
                            'predictions': predictions},
                            CFG.OUTPUT_DIR + f"/{CFG.MODEL.replace('/', '_')}_fold_{i}_best.pth")
        torch.cuda.empty_cache()
        gc.collect()
