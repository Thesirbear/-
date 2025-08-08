from Dataparser.Audio_dataset import Audio_dataset
from torch.utils.data import DataLoader


def Get_Data(transformation, args, path_to_La):
    train_dataset = Audio_dataset( 
        root_dir = path_to_La, 
        dataset_type = 'train',
        access_type = 'LA',
        technology_type = 'CM',
        transformation = transformation,
        args = args
    )

    dev_dataset = Audio_dataset(
        root_dir = path_to_La,
        dataset_type = 'dev',
        access_type = 'LA',
        technology_type = 'CM',
        transformation = transformation,
        args = args,
    )

    eval_dataset = Audio_dataset(
        root_dir = path_to_La,
        dataset_type = 'eval',
        access_type = 'LA',
        technology_type = 'CM',
        transformation = transformation,
        args = args,
    )

    train_dataloader = DataLoader(
        dataset = train_dataset, 
        batch_size=args.batch_size, 
        shuffle = True,
        # num_workers = args.num_workers,
    )

    test_dataloader = DataLoader(
        dataset = dev_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        # num_workers = args.num_workers,
        pin_memory = True,
    )

    eval_dataloader = DataLoader(
        dataset = eval_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        # num_workers = args.num_workers,
    )

    return train_dataloader, test_dataloader, eval_dataloader
