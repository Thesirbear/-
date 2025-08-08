
import wandb
import torch

def train_one_epoch(train_dataloader, model, optimizer, scheduler, epoch_step, criterion, args):
    model.train()

    avg_loss = 0
    step = epoch_step * len(train_dataloader)
    for batch_idx, (audio, label) in enumerate(train_dataloader):
        
        audio, label = audio.to(args.device), label.to(args.device)

        output = model(audio)
        loss = criterion(output, label)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


        avg_loss += loss.item()

        wandb.log({
            "train_step_loss": loss.item(),
            "lr": scheduler.get_last_lr()[0]
        }, step = step + batch_idx)


    avg_loss = avg_loss / (batch_idx + 1)
    return avg_loss