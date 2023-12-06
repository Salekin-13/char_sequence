import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from metadata import char as metadata

DATA_DIR = metadata.DIRECTORY
MAX_LENGTH = metadata.MAX_LENGTH

BATCH_SIZE = 32

class BanglaChar():
    def __init__(
        self,
        datatype,
        max_len = None,
        with_start_end_token = False,
        ):
        super().__init__()
        self.max_length = MAX_LENGTH if max_len is None else max_len
        self.start_end = with_start_end_token
        
        self.mapping = metadata.MAPPING
        self.output_dims = (self.max_length, 1)
        
        max_width = metadata.IMAGE_WIDTH #* self.max_length
        input_dim = (*metadata.INPUT_DIM[:2],max_width)
        self.resize = input_dim[1:]
        
        csv_path = f'{DATA_DIR}/{datatype}.csv' 
        df = pd.read_csv(csv_path, encoding='utf-8')
        df = df[['image_id', 'target']]
        self.image_ids = df.image_id.values
        self.image_paths = [f'{DATA_DIR}/{datatype}/{image_id}.jpg' for image_id in self.image_ids]
        self.target = df.target.values
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) 
        ])
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self,item):
        image = Image.open(self.image_paths[item])
        
        if metadata.IMAGE_CHANNEL == 3:
            image = image.convert('RGB')
        elif metadata.IMAGE_CHANNEL == 1:
            image = image.convert('L')    
        
        
        image = image.resize((self.resize[1],self.resize[0]))        
        image = self.transform(image)
        
        target = self.target[item]
        label = convert_strings_to_labels(
                target,
                self.mapping,
                length=self.output_dims[0],
                with_start_end_tokens=self.start_end,
            )
        
        return {
            "images": image.to(dtype = torch.float),
            "targets": torch.tensor(label,dtype = torch.long)
        }
        
def dividing_datasets(with_start_end_token = None):
    
    with_start_end_token = with_start_end_token if with_start_end_token is not None else False
    
    train = BanglaChar(datatype='train',with_start_end_token=with_start_end_token)
    test = BanglaChar(datatype='test',with_start_end_token=with_start_end_token)
    val = BanglaChar(datatype='val',with_start_end_token=with_start_end_token)

    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, test_loader, val_loader        
        
        
def convert_strings_to_labels(
    word, mapping, length, with_start_end_tokens
):
    word = [char for char in word if char not in ['\u200c', '\"', '-', '\u200d', '\n', '\xa0', '\r']]
    if with_start_end_tokens:
        word = ['<S>', *word, '<E>']
    labels = [mapping.index(char) for char in word]  
    labels = labels + [3]*(length - len(labels))
    return labels            
    
    
    
        