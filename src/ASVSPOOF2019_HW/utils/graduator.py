from Dataparser.Audio_dataset_grading import Audio_dataset_grading
from torch.utils.data import DataLoader
import torchaudio
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from ASVSPOOF2019_HW.utils.grade.grading import grade

from Model.LCNN import LCNN


def graduate(args, transformation, path_to_model_weights):
    eval_dataset = Audio_dataset_grading(
        root_dir = args.data_dir,
        dataset_type = 'eval',
        access_type = 'LA',
        technology_type = 'CM',
        transformation = transformation,
        args = args,
    )

    eval_dataloader_grading = DataLoader(
        dataset = eval_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        # num_workers = args.num_workers,
    )

    model = LCNN(
        input_channels = args.input,
        hidden_channels = args.hidden_dim,
        num_of_samples = args.num_samples,
        output_size = args.output
    )
    model.load_state_dict(torch.load(path_to_model_weights, weights_only=True))
    model.eval()

    name_res = []
    audio_res = []
    for batch_idx, (audio, label) in enumerate(eval_dataloader_grading):
        audio = audio.to(args.device)
        name, label = label
        label.to(args.device)

        output = model(audio)
        audio_res.append(output)
        name_res.append(name)

    data = dict({"key": name_res, "score": audio_res})
    data = pd.DataFrame(data)

    data.to_csv(os.path.join(args.solution_path, "vagrigorzhevskii@edu.hse.ru.csv"))
    os.system("python" + args.grade_path)

