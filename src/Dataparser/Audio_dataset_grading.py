import torchaudio
import torch
from torch.utils.data import Dataset
import pandas as pd
import os


class Audio_dataset_grading(Dataset):
    
    def _get_annot_path_(self):
        path = (self.root_dir +
            self.technology_type.lower() + '_protocols/' +
            self.special_name[:-1] + '.' + self.access_type + '.' +
            self.technology_type.lower() + '.' + self.dataset_type)
        
        #By dataset type define right way to the annot_path
        if (self.dataset_type == 'train'):
            path += '.trn.txt'
        elif (self.dataset_type == 'dev' or self.dataset_type == 'eval'):
            path += '.trl.txt'
        else:
            raise 'Undefined dataset_type'
        return path
    
    def _define_short_name(self):
        if (self.dataset_type == 'dev'):
            self.dataset_short_name = 'D'
        elif (self.dataset_type == 'eval'):
            self.dataset_short_name = 'E'
        elif (self.dataset_type == 'train'):
            self.dataset_short_name = 'T'
        if (self.dataset_short_name == 'NaN'):
            raise "Type of dataset not found!"
    
    def __init__(self, root_dir, dataset_type, access_type, technology_type, transformation, args):
        super(Audio_dataset_grading, self).__init__()

        self.special_name = 'ASVspoof2019_'
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.access_type = access_type
        self.technology_type = technology_type
        self.num_of_samples = args.num_samples

        self.device = args.device

        self.transformation = transformation

        #Parsing path
        self.annot_path = self._get_annot_path_()

        self.annotation = pd.read_csv(self.annot_path, sep=' ')

        #Modifing datatable to make it look nicer
        if (self.access_type == 'LA'):
            if (self.technology_type == 'CM'):
                self.annotation = self.annotation.drop(self.annotation.columns[[0, 2, 4]], axis = 1)
            else:
                pass
        else:
            pass

        self.dataset_short_name = 'NaN'
        #Defining a special short name for getitem func
        self._define_short_name()
    
    def __len__(self):
        return len(self.annotation)
    
    def _get_typecode(self, audio_type):
        if (self.technology_type == 'CM'):
            if (audio_type == '-'): #Here we define, bonafide speech
                audio_type = 1.0
            else:
                audio_type = 0.0
        return audio_type
    
    def _normalize_lenght_of_audio_(self, audio):
        if (audio.shape[2] > self.num_of_samples):
            audio, x = torch.split(audio, [self.num_of_samples, audio.shape[2] - self.num_of_samples], dim = 2)

        if (audio.shape[2] < self.num_of_samples):
            difference = self.num_of_samples - audio.shape[2]
            audio = torch.nn.functional.pad(audio, (0, difference))
        return audio

    def __getitem__(self, index):

        #Parsing by index name of the file and it's type
        audio_name, audio_type = self.annotation.iloc[index].tolist()

        audio_path = os.path.join(
            str(self.root_dir + #Prefix path to directories
            self.dataset_type), #Exact path files of our dataset
            "flac",
            str(audio_name + ".flac")
        )

        #Getting our audio_type (int) between 0 to 19
        # Where 0 - bonafide 
        # And [1, 19] - is different types of spoofing attacks
        audio_type = self._get_typecode(audio_type)

        #Loading audio and performing transofrmation
        audio, pr = torchaudio.load(audio_path)
        audio = self.transformation(audio)
        audio = self._normalize_lenght_of_audio_(audio)

        return (audio, (audio_name, audio_type))
        