"""An example file to run an experiment.
Keep this, it's used as an example to run the code after a user installs the project.
"""

import logging
import os
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig

#Torch imports
import torch, torchaudio


from ASVSPOOF2019_HW import utils
from Dataparser.Audio_dataloader import Get_Data
from Training import Run
from Model.LCNN import LCNN

# Refers to utils for a description of resolvers
utils.config.register_resolvers()

# Hydra sets up the logger automatically.
# https://hydra.cc/docs/tutorials/basic/running_your_app/logging/
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(config: DictConfig) -> None:
    # Using the template provides utilities for experiments:

    # 1. Setting up experiment and resuming directories

    utils.config.setup_wandb(config)
    utils.seeding.seed_everything(config.seed)
    #Manage device
    config.device = "cuda" if torch.cuda.is_available() else "cpu"

    #Model creation
    model = LCNN(
        input_channels = config.input,
        hidden_channels = config.hidden_dim, 
        output_size = config.output,
        num_of_samples = config.num_samples
    )
    model.to(config.device);


    #Our front-end for Audio proccesing
    transformation = torchaudio.transforms.LFCC(
        sample_rate = config.sample_rate,
        n_lfcc = config.lfcc_coef,
        n_filter = config.lfcc_filters,
    )

    #Loading data
    train_dataloader, test_dataloader, eval_dataloader = Get_Data(
        transformation = transformation,
        path_to_La = config.data_dir,
        args = config
    )
    
    #Creating all important stuff
    criterion = torch.nn.CrossEntropyLoss().to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max = config.epochs * len(train_dataloader),
    eta_min = config.eps_adamw)


    #Training our model
    with wandb.init(
            project = "ASVspoof-2019",
            name = "Check-run"
        ) as run: 
        Run(
            train_dataloader = train_dataloader, 
            test_dataloader = test_dataloader, 
            eval_dataloader = eval_dataloader,
            model = model, 
            optimizer = optimizer, 
            scheduler = scheduler, 
            criterion = criterion, 
            args = config
        )


    #Making final evaluation by grading.py

    #Evaluation of best model
    graduate("")

if __name__ == "__main__":
    main()