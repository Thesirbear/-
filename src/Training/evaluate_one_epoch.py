import torch
import numpy as np
from ASVSPOOF2019_HW.utils.calculate_eer import compute_eer

def evaluate_one_epoch(test_dataloader, eval_dataloader, model, criterion, args):
    model.eval()

    #variables for dev
    avg_eval_loss = 0
    accuracy = 0
    #variables for eval
    
    with torch.no_grad():
        total_elems = 0
        for batch_idx, (audio, label) in enumerate(test_dataloader):
            audio, label = audio.to(args.device), label.to(args.device)
    
            output = model(audio)
            loss = criterion(output, label) # criterion is loss function
            # here we particularly check our answer with original one

            accuracy += (output.argmax(-1) == label).sum().item()
            avg_eval_loss += loss.item()
            total_elems += output.shape[0]
        bona_cm = []
        spoof_cm = []
        for batch_idx, (audio, label) in enumerate(eval_dataloader):
            audio, label = audio.to(args.device), label.to(args.device)

            output = model(audio)
            bona_cm_batch = []
            spoof_cm_batch = []
            for x in range(label.shape[0]):
                if (label[x] == 1):
                    bona_cm_batch.append(output[x].item())
                else:
                    spoof_cm_batch.append(output[x].item())
            bona_cm_batch = np.array(bona_cm_batch).astype(np.float64)
            spoof_cm_batch = np.array(spoof_cm_batch).astype(np.float64)
            bona_cm = np.concatenate((bona_cm, bona_cm_batch))
            spoof_cm = np.concatenate((spoof_cm, spoof_cm_batch))
    
    eer_eval = compute_eer(bona_cm, spoof_cm)[0]
    avg_eval_loss = avg_eval_loss / (batch_idx + 1)
    accuracy = 100 * accuracy / total_elems
    eer_eval *= 100
    return avg_eval_loss, accuracy, eer_eval