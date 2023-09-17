import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from network import GenreClassification, evaluate
import torch
import yaml
import argparse
import os
from resnet import *
torch.cuda.empty_cache()

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD, save_path=None):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        print(epoch)
        for batch in train_loader:
            loss = model.training_step(batch, args.use_GPU)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

        if save_path:
            os.makedirs(save_path, exist_ok = True)
            torch.save(model.state_dict(), save_path.format(epoch=epoch+1))

    return history


    
def main(args):
    with open(args.config, 'rb') as f:
        config=yaml.safe_load(f.read())
    
    if args.use_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU_id)
    
    train_tfms = tt.Compose([tt.Resize((224, 224), ), tt.ToTensor()])
    valid_tfms = tt.Compose([tt.ToTensor()])
    

    dataset = ImageFolder(config["dataset"], train_tfms)

    batch_size = config["batch_size"]
    val_size = 2000
    train_size = len(dataset) - val_size 

    train_data,val_data = random_split(dataset,[train_size,val_size])
    train_dl = DataLoader(train_data, batch_size, shuffle = True, pin_memory = True)
    val_dl = DataLoader(val_data, batch_size*2, pin_memory = True)
    
    save_path =config["save_folder"]
    model = ConvNet()
    
    if args.use_GPU:
        model = model.cuda()
    history = fit(config["epoch"], config["lr"], model, train_dl, val_dl, opt_func=torch.optim.Adam, save_path=save_path)
    

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description="Training a CNN on Style Classification",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", default="config.yaml", help="config file in the yaml format")
    parser.add_argument("--use_GPU",action="store_true", default=False, help="Do you want to use GPU or not")
    parser.add_argument("--GPU_id", default=0, help="Select GPU number if there is multiple ones")
    args=parser.parse_args()   
     
    main(args)