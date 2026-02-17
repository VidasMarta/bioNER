import json
import os
from preprocessing import Embedding
import settings
import torch

MAX_LEN = 256

class DatasetLoader:
    def __init__(self, dataset_name, dataset_path):
        self.dataset_path = dataset_path
        self.folder_path = os.path.join(dataset_path, dataset_name)
        self.dataset_name = dataset_name
    
    def _load_file(self, file_name): 
        file_path = os.path.join(self.folder_path, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                text, tags = [], []
                for line in f:
                    line = json.loads(line)
                    text.append(line["tokens"])
                    tags.append(line["tags"])
            return text, tags 
        else:
            print(f"File {file_path} not found")

    def load_data(self, train_filepath = "train.json" ):
        ''' 
        Load the dataset
        Returns:    
            total_tags: List of all tags in the dataset and their encoding
            (text_train, tags_train): List of training data split into text and tags
            (text_val, tags_val): List of validation data split into text and tags
            (text_test, tags_test): List of testing data split into text and tags  
        '''
        total_tags = {}
        text_train, text_val, text_test = [], [], []
        tags_train, tags_val, tags_test = [], [], []

        if len(train_filepath.split('/')) == 1:
            train_filepath = os.path.join(self.folder_path, train_filepath)
        else:
            train_filepath = os.path.join(self.dataset_path, train_filepath)

        tags_file = os.path.join(self.folder_path, "label.json")        
        train_file = train_filepath
        val_file = os.path.join(self.folder_path, "devel.json")
        test_file = os.path.join(self.folder_path, "test.json")
        
        with open(tags_file, "r", encoding="utf-8") as f:
                total_tags = json.load(f)
        self.num_tags = len(total_tags)
        text_train, tags_train = self._load_file(train_file)
        text_val, tags_val =self._load_file(val_file)
        text_test, tags_test = self._load_file(test_file)

        return total_tags, (text_train, tags_train), (text_val, tags_val), (text_test, tags_test)


class Dataset:
    def __init__(self, tokens, tags, attention_masks, crf_masks):
        self.tokens = tokens
        self.tags = tags
        self.attention_masks = attention_masks
        self.crf_masks = crf_masks


    def __len__(self):
        return len(self.tokens) 
    

    def __getitem__(self, idx):
        return (
            self.tokens[idx], 
            self.tags[idx],  
            self.attention_masks[idx],
            self.crf_masks[idx]
        )

def get_max_len(text_train): 
    max_len = max(len(sublist) for sublist in text_train)
    return min(max_len, MAX_LEN)
 


    