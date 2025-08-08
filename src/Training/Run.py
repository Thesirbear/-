from Training.train_one_epoch import train_one_epoch
from Training.evaluate_one_epoch import evaluate_one_epoch
import wandb
import torch
import os

def Run(train_dataloader, test_dataloader, eval_dataloader, args, model, optimizer, scheduler, criterion):
    best_model = model.state_dict(), generally_last_one = model.state_dict()
    best_eer = 1000000
    for epoch_step in range(args.epochs):
        avg_loss = train_one_epoch(
            train_dataloader = train_dataloader, 
            model = model, 
            optimizer = optimizer,
            scheduler = scheduler,
            epoch_step = epoch_step, 
            criterion = criterion, 
            args = args
        )

        if (epoch_step % 5 == 0):
            avg_eval_loss, accuracy, eer = evaluate_one_epoch(
                test_dataloader = test_dataloader,
                eval_dataloader = eval_dataloader,
                model = model, 
                criterion = criterion, 
                args = args
            )
            wandb.log({
                "train_avg_loss": avg_loss,
                "val_avg_loss": avg_eval_loss,
                "val_accuracy": accuracy,
                "err": eer,
            }, step = (epoch_step + 1) * len(train_dataloader))
            # print(accuracy)
        else:
            wandb.log({
                "train_avg_loss": avg_loss,
            }, step = (epoch_step + 1) * len(train_dataloader))
        generally_last_one = model.state_dict()
        if (min(eer, best_eer) == eer):
            best_model = generally_last_one
    torch.save(best_model, os.path.join(args.output_dir, "model_best.pth"))
    torch.save(generally_last_one, os.path.join(args.output_dir, "last_model.pth"))
