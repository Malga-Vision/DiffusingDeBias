
import torch
import torchvision
import pandas as pd
import os
from tqdm import tqdm
from utils.metrics import *

PATH_TO_MODELS = "saved_models"


def train_model_erm(model: torch.nn.Module, train_loader, val_loader, test_loader, device, optimizer: torch.optim.Optimizer, num_classes, num_biases = 1, epochs=10, wb=None, make_figures=False, name="jtt"):
    cur_model_name = f"{name}-init.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS,cur_model_name))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for epoch in range(epochs):
        loss_task_tot = AverageMeter()
        top1 = AverageMeter()
        
        subgroups_shape = (num_classes,) * (1+num_biases) #if num_classes <= 2 else (num_classes, 2)

        subgroup_top1 = AverageMeterSubgroups(subgroups_shape, device=device)
        tk0 = tqdm(
            train_loader, total=int(len(train_loader)), leave=True, dynamic_ncols=True
        )

        epoch_outputs = []
        epoch_targets = []
        epoch_biases  = []
        model.train()
        with torch.enable_grad():
            for batch, (dat, labels, _) in enumerate(tk0):
                dat = dat.to(device)
                target = labels[0].to(device).long()
                bias_l = labels[1].to(device).long()
                
                with torch.cuda.amp.autocast(enabled=True):
                    output = model(dat)
                    if len(labels) > 2:
                        labels = torch.vstack(labels)
                    
                    if make_figures:                    
                        epoch_outputs.append(output)
                        epoch_targets.append(target)
                        epoch_biases.append(bias_l)
                
                    loss_task: torch.Tensor = model.loss_fn(output, target)
                    
                loss_task_tot.update(loss_task.item(), dat.size(0))
                scaler.scale(loss_task).backward()
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                optimizer.zero_grad()

                acc1  = accuracy(output, target, topk=(1,))
                acc_a = regroup_by(subgroup_top1, ("aligned",))
                acc_c = regroup_by(subgroup_top1, ("misaligned",))

                subgroup_masks = get_subgroup_masks(labels, num_classes=subgroups_shape, device=device)
                subgroup_acc1 = accuracy_subgroup(output, target, subgroup_masks, num_classes=num_classes)
                
                top1.update(acc1[0], dat.size(0))
                subgroup_top1.update(subgroup_acc1, subgroup_masks)

                postifix_dict = {
                    "epoch": epoch,
                    "acc1": top1.avg,
                    "lr": optimizer.param_groups[0]['lr'],
                    "acc_a": acc_a[0].item(),
                    "acc_c": acc_c[0].item()
                }                    
                subgroup_avg = subgroup_top1.avg
                
                if len(subgroup_top1.avg.size()) > 2:
                    for cl in range(subgroup_avg.size(0)):
                        for b0 in range(subgroup_avg.size(1)):
                            for b1 in range(subgroup_avg.size(2)):
                                value = subgroup_avg[cl, b0, b1].item()
                                if value <= 0:
                                    continue
                                postifix_dict[f"({cl},{b0},{b1})"] = value
                else:
                    for cl in range(subgroup_avg.size(0)):
                        for g in range(subgroup_avg.size(1)):
                            value = subgroup_avg[cl, g].item()
                            if value <= 0: 
                                continue
                            postifix_dict[f"({cl},{g})"] = value
                postifix_dict["loss"] = loss_task_tot.avg

                iter_string = f"Training Set Epoch {epoch} (iter {(epoch+1) * batch}): \n"
                for key in postifix_dict.keys():
                    iter_string += f"{key}:\t {postifix_dict[key]}\n"                        

                # tk0.write(iter_string)
                tk0.set_postfix(postifix_dict)
                
                torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))
        
        if wb is not None:
            wb.log_output(postifix_dict)

        if val_loader is not None:
            evaluate_model(
                model, 
                val_loader, 
                num_classes, 
                num_biases=num_biases, 
                criterion=torch.nn.CrossEntropyLoss(),
                epoch=epoch,
                device=device,
                wb=wb,
                prefix="val"
            )

        if test_loader is not None:        
            evaluate_model(
                model, 
                test_loader, 
                num_classes, 
                num_biases=num_biases, 
                criterion=torch.nn.CrossEntropyLoss(),
                epoch=epoch,
                device=device,
                wb=wb,
                prefix="test"
            )        

    cur_model_name = f"{name}-final.pt"
    print(f"Saving {cur_model_name}")
    torch.save(model.state_dict(), os.path.join(PATH_TO_MODELS, cur_model_name))
    return cur_model_name

@torch.no_grad()
def evaluate_model(model, test_loader, num_classes, num_biases, criterion, epoch, device, wb, prefix="val"):
    model.eval()
    groups_size = (num_classes,) * (1+num_biases)
    g_criterion = torch.nn.CrossEntropyLoss(reduction="none")
    loss_task_tot   : AverageMeter = AverageMeter()
    top1            : AverageMeter = AverageMeter()
    subgroup_top1   : AverageMeterSubgroups = AverageMeterSubgroups(
        size   = groups_size, 
        device = device
    )

    # subgroup_loss  : AverageMeterSubgroups = AverageMeterSubgroups(size=groups_size, device=device)
    
    tk0 = tqdm(
        test_loader, total=int(len(test_loader)), leave=True, dynamic_ncols=True
    )
    
    for batch, (dat, labels, _) in enumerate(tk0):
        dat     : torch.Tensor = dat.to(device)
        target  : torch.Tensor = labels[0].to(device)
        bias_t  : torch.Tensor = labels[1].to(device)
        output  : torch.Tensor = model(dat)
        if len(labels) > 2:
            labels = torch.vstack(labels)
                
        loss    : torch.Tensor = criterion(output, target)   
        gloss   : torch.Tensor = g_criterion(output, target)     
        loss_task_tot.update(loss.item(), dat.size(0))
        
        acc1 = accuracy(output, target, topk=(1, ))
        subgroup_masks = get_subgroup_masks(
            labels = labels, 
            num_classes = groups_size, 
            device = device
        )
        subgroup_acc1 = accuracy_subgroup(output, target, subgroup_masks, num_classes=num_classes)
        # sg_loss = loss_subgroup(gloss, subgroup_masks)       

        
        top1.update(acc1[0], dat.size(0))
        subgroup_top1.update(subgroup_acc1, subgroup_masks)
        # subgroup_loss.update(sg_loss, subgroup_masks)

        acc1  = top1.avg
        avg_loss = loss_task_tot.avg
        # wg_loss = sg_loss.min()

        acc_a = regroup_by(subgroup_top1, ("aligned", ))
        acc_c = regroup_by(subgroup_top1, ("misaligned",))

        postifix_dict = {
            "epoch": epoch,
            f"{prefix}_acc1": top1.avg,
            f"{prefix}_acc_a": acc_a[0].item(),
            f"{prefix}_acc_c": acc_c[0].item()
        }
        subgroup_avg = subgroup_top1.avg
        
        if len(subgroup_top1.avg.size()) > 2:
            for cl in range(subgroup_avg.size(0)):
                for b0 in range(subgroup_avg.size(1)):
                    for b1 in range(subgroup_avg.size(2)):
                        value = subgroup_avg[cl, b0, b1].item()
                        if value <= 0:
                            continue
                        postifix_dict[f"{prefix}-({4*cl+2*b0+b1})"] = value
        else:
            for cl in range(subgroup_avg.size(0)):
                for g in range(subgroup_avg.size(1)):
                    value = subgroup_avg[cl, g].item()
                    if value <= 0: 
                        continue
                    postifix_dict[f"{prefix}-({cl},{g})"] = value
        postifix_dict[f"{prefix}_loss"] = loss_task_tot.avg
        
        iter_string = f"{prefix} Set Epoch {epoch}: \n"
        for key in postifix_dict.keys():
            iter_string += f"{key}:\t {postifix_dict[key]}\n"                        

        # tk0.write(iter_string)
        tk0.set_postfix(postifix_dict)
        
    if wb is not None:
        wb.log_output(postifix_dict)

    return avg_loss, None
