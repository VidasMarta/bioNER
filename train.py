import argparse
import os
import random
import models
from preprocessing import Embedding, CharEmbeddingCNN
from datasets import Dataset
from evaluation import Evaluation
import torch
from torch.optim import *
import settings
from datasets import *
import numpy as np
from utils.logger import Logger
from utils import trainer
import copy

from utils.extract_wrong_words import evaluate_and_find_errors

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, required=False, help='Path to the config file', default='/lustre/home/mvidas/Diplomski/experiments/default_train.yml')    
    parser.add_argument('--model_name', type=str, required=False, help='Name of the file used for saving model weights', default='bilstm_crf')    
    return parser.parse_args()

def count_parameters(model):
    model.bert
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(model_name, model_args, settings_args, logger):    
    if torch.cuda.is_available():
        device = 'cuda'
        print("Using cuda")
    else:
        device = 'cpu'
        print("Using cpu")
    
    dataset_loader = DatasetLoader(settings_args['dataset'], settings.DATA_PATH)
    tag_to_num, (text_train, tags_train), (text_val, tags_val), (text_test, tags_test) = dataset_loader.load_data(settings_args['train_filename'])
    num_tags = len(tag_to_num) #number of tags used (e.g. 3 for O, B-entity, I-entity)
    num_to_tag = dict((v,k) for k,v in tag_to_num.items()) 
    
    max_len = model_args['max_length'] 
    word_embedding = settings_args['word_embedding']
    word_embeddings_model = Embedding.create(word_embedding, dataset_loader.dataset_name, max_len) 

    eval = Evaluation(word_embeddings_model, settings_args["tagging_scheme"])

    #if no char cnn is used
    char_emb = None
    test_char_embeddings = None
    model_args['char_embedding_dim'] = None

    batch_size = model_args['batch_size']
    if settings_args['char_cnn_embedding']:
        vocab = settings_args['cnn_vocab']
        char_emb_size = settings_args['cnn_embedding_dim']
        model_args['char_embedding_dim'] = char_emb_size
        feature_size = settings_args['feature_size']
        max_word_len = settings_args['cnn_max_word_len']
        char_emb = CharEmbeddingCNN(vocab, char_emb_size, feature_size, max_word_len)
        test_char_embeddings = char_emb.batch_cnn_embedding_generator(text_test, max_len, batch_size)
        
    
    tokens_train_padded, tags_train_padded, attention_masks_train, crf_mask_train = word_embeddings_model.tokenize_and_pad_text(text_train, tags_train)
    train_data = Dataset(tokens_train_padded, tags_train_padded, attention_masks_train, crf_mask_train)
    tokens_val_padded, tags_val_padded, attention_masks_val, crf_mask_val = word_embeddings_model.tokenize_and_pad_text(text_val, tags_val)
    val_data = Dataset(tokens_val_padded, tags_val_padded, attention_masks_val, crf_mask_val)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    valid_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    tokens_test_padded, tags_test_padded, attention_masks_test, crf_mask_test= word_embeddings_model.tokenize_and_pad_text(text_test, tags_test)
    test_data = Dataset(tokens_test_padded, tags_test_padded, attention_masks_test, crf_mask_test)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    model_args['word_embedding'] = settings_args['word_embedding']

    if settings_args['bert_finetuning'] and "bert" in settings_args['word_embedding'].lower():
        model_args['ft_lr'] = settings_args['ft_lr']
        trainer_object = trainer.Finetuning_Trainer(model_name, model_args, num_tags, train_data_loader, valid_data_loader, 
        word_embeddings_model, char_emb, text_train, text_val, max_len, batch_size, 
        device, num_to_tag, eval, logger)
        print("Number of trainable parameters in a NER model: ", count_parameters(trainer_object.model))    
        trainer_object.train()

        best_model_weights = torch.load(settings.MODEL_PATH + f"/{model_name}_best.bin")
        best_model = models.ft_bb_BiRNN_CRF(num_tags, model_args, model_args['char_embedding_dim']) 
        best_model.load_state_dict(best_model_weights)
        print("Testing")
        eval.evaluate(test_data_loader, best_model, device, test_char_embeddings, num_to_tag, logger, True)

    else:
        trainer_object = trainer.Normal_Trainer(model_name, model_args, num_tags, train_data_loader, valid_data_loader, 
        word_embeddings_model, char_emb, text_train, text_val, max_len, batch_size, 
        device, num_to_tag, eval, logger)
        print("Number of trainable parameters in a NER model: ", count_parameters(trainer_object.model))
        trainer_object.train()

        best_model_weights = torch.load(settings.MODEL_PATH + f"/{model_name}_best.bin")
        best_model = models.BiRNN_CRF(num_tags, model_args, word_embeddings_model.embedding_dim, model_args['char_embedding_dim'])
        best_model.load_state_dict(best_model_weights)
        print("Testing")
        eval.evaluate(test_data_loader, best_model, device, test_char_embeddings, num_to_tag, logger)
        

def set_seed(seed: int = 42): ##za reproducility, izvor: https://medium.com/we-talk-data/how-to-set-random-seeds-in-pytorch-and-tensorflow-89c5f8e80ce4
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def extract_args():
    args = parse_args()
    model_name = args.model_name
    model_args, settings_args = settings.Settings(args.config)
    return model_name, model_args, settings_args

def print_args(model_args, settings_args):
    print("Training with the following settings:")
    print("Model Args:", model_args)
    print("Settings Args:", settings_args)

if __name__ == "__main__":    
    #---> Train normal with 5 different seeds
    '''model_name, model_args, settings_args = extract_args()
    settings_args['dataset'] = 'distemist/synthetic/kshot_10pct'
    settings_args['train_filename'] = '' #train_10pct.json
    print_args(model_args, settings_args)
    output_path = os.path.join(settings.LOG_PATH, model_name)
    logger = Logger(output_path, model_args, settings_args)
    model_args['weights'] = None
    for seed in [42, 198, 6000, 3828, 7382]:
        set_seed(seed)
        new_model_name = model_name + f"_seed{seed}"
        main(new_model_name, model_args, settings_args, logger)
        
    #trainig_set - treba biti koje ime će u tablici taj model imati
    #file_to_save grupirati na koliko pct dataseta se trenira ili finetune-a model
    logger.calculate_mean_stddev(training_set="-", file_to_save=os.path.join(settings.LOG_PATH, "distemist/10_pct/test.csv"))'''
    """

    #---> Extract wrongly classified entities from validation and test sets (modelName = "D1_ftB_C_L_5_A_with_gen2Data")
    errors_val, errors_test= evaluate_and_find_errors(model_name, settings_args, model_args, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    

    with open(output_path+'/wrong_val.log',"w") as f:
        for error in errors_val:
            f.write(f"{error} \n")
    

    with open(output_path+'/wrong_test.log',"w") as f:
        for error in errors_test:
            f.write(f"{error} \n")"""

    #---> Train with generated data (with 5 different seeds - commented)
    model_name, model_args, settings_args = extract_args()
    model_name = "ES_S_with_0shot"
    settings_args['dataset'] = 'distemist/trf/synthetic/0shot/'
    settings_args['train_filename'] = 'corrected_generated_sentences_20260324.jsonl'
    print_args(model_args, settings_args)
    output_path = os.path.join(settings.LOG_PATH, model_name)
    logger = Logger(output_path, model_args, settings_args)

    new_model_name = "0shot_ft_with_10DISTEMIST" #D1_ftB_C_L_5_A_mean_with_genData
    output_path = os.path.join(settings.LOG_PATH, new_model_name)
    logger_2 = Logger(output_path)

    for seed in [42, 198, 6000, 3828, 7382]:
        set_seed(seed) 
        model_name_seed = model_name + f"_seed{seed}" 
        if not os.path.exists(settings.MODEL_PATH + f"/{model_name_seed}_best.bin"):
            model_args['weights'] = None
            main(model_name_seed, model_args, settings_args, logger)
        else:
            print("Ima predtreniranog")

        #Then take weights and finetune on NCBI-disease train set
        model_args_2 = copy.deepcopy(model_args)
        model_args_2['weights'] = torch.load(settings.MODEL_PATH + f"/{model_name_seed}_best.bin")
        settings_args_2 = copy.deepcopy(settings_args)
        settings_args_2['dataset'] = 'distemist/trf/'
        settings_args_2['train_filename'] = 'train_10pct.json' # 'train.json', train_50pct.json

        main(new_model_name, model_args_2, settings_args_2, logger_2)
        
    logger.calculate_mean_stddev(training_set="0-shot", file_to_save=os.path.join(settings.LOG_PATH, "distemist/0shot/test.csv"))
    logger_2.calculate_mean_stddev(training_set="0-shot", file_to_save=os.path.join(settings.LOG_PATH, "distemist/10_pct/test.csv"))
    


    
    