import torch
from transformers import AutoTokenizer, AutoModel
from allennlp.modules.elmo import Elmo, batch_to_ids
from abc import ABC, abstractmethod
import os
import settings
import torch.nn as nn

class Embedding(ABC): #For word embeddings
    def __init__(self, embedding_model_name, dataset_name, max_len=256):
        self.dataset_name = dataset_name
        self.max_len = max_len
        self.embedding_model_name = embedding_model_name
        self.embedding_dim = None
        self.crf_attention_mask = None

    @staticmethod
    def create(embedding_model_name, dataset_name, max_len=256):
        '''
        Factory method to create an embedding model
        Args:
            embedding_model_name: Name of the embedding model
            embeddings_path: Path to save embeddings
            dataset_name: Name of the dataset
            max_len: Maximum length of the input sequence
        Returns:
            Embedding model instance or raises ValueError if the embedding model is not supported
        '''
        if embedding_model_name.lower() == 'biobert':
            return Embedding_bioBERT(embedding_model_name, dataset_name, max_len)
        elif embedding_model_name.lower() == 'bioelmo':
            return Embedding_bioELMo(embedding_model_name, dataset_name, max_len)
        else:
            raise ValueError(f"Embedding {embedding_model_name} not supported, bioBERT and bioELMo are.")
        
    def _pad_or_truncate(self, seq, pad, pad_value):
        if seq.size(0) < self.max_len:
            return nn.functional.pad(input=seq, pad=pad, value=pad_value)
        else:
            return seq[:self.max_len]
        
    @abstractmethod
    def get_embedding(self, tokens):
        pass

    @abstractmethod
    def tokenize_and_pad_text(self, text, tags):
        pass

    @abstractmethod
    def get_relevant_tags(self, tags, num_to_tag_dict, word_level_masks):
        pass

    @abstractmethod
    def get_word_on_index(self, tokens, idxs):
        pass
    
#bioLinkBERT https://huggingface.co/michiyasunaga/BioLinkBERT-base/tree/main
#bioLinkBERTlarge https://huggingface.co/michiyasunaga/BioLinkBERT-large/tree/main TODO: change self.embedding dim to 1024
class Embedding_bioBERT(Embedding): #bioBERT large https://github.com/dmis-lab/biobert?tab=readme-ov-file TODO: change self.embedding dim to 1024
    def __init__(self, embedding_model_name, dataset_name, max_len=256):
        super(Embedding_bioBERT, self).__init__(embedding_model_name, dataset_name, max_len)
        self.max_len = max_len
        self.embedding_dim = 1024  # Dimensionality of BioBERT embeddings 768, for large 1024
        bioBERT_setup_path = os.path.join(settings.EMBEDDINGS_PATH, "bioLinkBERTlarge_setup") 
        self.tokenizer = AutoTokenizer.from_pretrained(bioBERT_setup_path)
        self.bert = AutoModel.from_pretrained(bioBERT_setup_path)

    def tokenize_and_pad_text(self, text, tags):
        all_input_ids = []
        all_padded_tags = []
        all_attention_masks = []
        all_word_level_masks = [] #needed for crf

        encoding = self.tokenizer(
                text,
                is_split_into_words=True,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt',
                return_attention_mask=True,
            )

        for batch_idx, sentence_tags in enumerate(tags):
            input_ids = encoding["input_ids"][batch_idx]
            attention_mask = encoding["attention_mask"][batch_idx]
            word_ids = encoding.word_ids(batch_index=batch_idx)

            # Create word-level mask: 1 for first subword of each word
            word_level_mask = []
            aligned_tags = []
            prev_word_id = None
            # only first subword is labeled, others are labeled as -1 (https://datascience.stackexchange.com/questions/69640/what-should-be-the-labels-for-subword-tokens-in-bert-for-ner-task)
            for word_id in word_ids:
                if word_id is None: #padding or other special token (cls, sep)
                    word_level_mask.append(0)
                    aligned_tags.append(-1)
                elif word_id != prev_word_id: #first subword
                    word_level_mask.append(1)
                    aligned_tags.append(sentence_tags[word_id])
                    prev_word_id = word_id
                else: # other subword 
                    word_level_mask.append(0)
                    aligned_tags.append(-1)

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_padded_tags.append(torch.tensor(aligned_tags))
            all_word_level_masks.append(torch.tensor(word_level_mask))
        
        return torch.stack(all_input_ids), torch.stack(all_padded_tags), torch.stack(all_attention_masks), torch.stack(all_word_level_masks)


    def get_embedding(self, token_list, attention_masks): 
        self.bert.eval()
        with torch.no_grad():
            outputs = self.bert(token_list, attention_mask=attention_masks)
    
        return outputs.last_hidden_state

    def get_relevant_tags(self, tags, num_to_tag_dict, word_level_masks):
        all_relevant_tags = []
        for tag_seq, mask in zip(tags, word_level_masks):
            relevant_tags = []
            for tag, is_first_subword in zip(tag_seq, mask):
                if is_first_subword and tag != -1:
                    relevant_tags.append(num_to_tag_dict[int(tag)])
            all_relevant_tags.append(relevant_tags)
        return all_relevant_tags
    
    def get_word_on_index(self, tokens, idxs):
        tokens_of_interest = [tokens[i] for i in idxs]
        return self.tokenizer.decode(tokens_of_interest)

class Embedding_bioELMo(Embedding):
    def __init__(self, embedding_model_name, dataset_name, max_len=256):
        super(Embedding_bioELMo, self).__init__(embedding_model_name, dataset_name, max_len)
        # weights and options files downloaded from https://github.com/Andy-jqa/bioelmo?tab=readme-ov-file
        set_up_path = os.path.join(settings.EMBEDDINGS_PATH, "bioELMo_setup")
        options_file = os.path.join(set_up_path, "biomed_elmo_options.json")
        weight_file = os.path.join(set_up_path, "biomed_elmo_weights.hdf5")
        self.elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=0)
        self.embedding_dim = 1024  # Dimensionality of BioELMo embeddings
        self.max_len = max_len

        
    def tokenize_and_pad_text(self, text, tags): 
        sentences_list = [["".join(word) for word in line] for line in text]
        tokenized_text = batch_to_ids(sentences_list)

        tensor_tags = [torch.tensor(t) for t in tags] 

        tokens_padded = torch.stack([self._pad_or_truncate(seq, (0, 0, 0, self.max_len - seq.shape[0]), 0) for seq in tokenized_text])  #TODO  
        tags_padded = torch.stack([self._pad_or_truncate(seq, (0, self.max_len - seq.shape[0]), -1) for seq in tensor_tags])         

        padding_mask = torch.where(tags_padded != -1, 1, 0)  

        self.crf_attention_mask = padding_mask
        return tokens_padded, tags_padded, padding_mask, padding_mask 

    def get_embedding(self, tokens, attention_masks):  
        self.elmo.eval() 
        with torch.no_grad():
            elmo_output = self.elmo(tokens)

            embeddings = elmo_output["elmo_representations"][0] # Removing the first dimension since it is 1
            mask = elmo_output["mask"]
            
            embeddings = embeddings * mask.unsqueeze(-1)  # Apply mask along token dimension

            return embeddings 

    def get_relevant_tags(self, tags, num_to_tag_dict, word_level_masks):
        return [[num_to_tag_dict[int(tag)] for tag in seq if int(tag) != -1] for seq in tags]
    
    def get_word_on_index(self, tokens, idxs):
        return [tokens[i] for i in idxs]


class CharEmbeddingCNN(nn.Module): 
    def __init__(self, vocab, emb_size, feature_size, max_word_length, dropout=0.1): 
        super(CharEmbeddingCNN, self).__init__()
       
        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.unk_idx = self.char_to_idx.get("<UNK>", 0) 
        self.max_word_length = max_word_length
        self.embedding_dim = emb_size

        # Define deep CNN layers (inspired by https://github.com/ahmedbesbes/character-based-cnn/blob/master/src/model.py)
        self.dropout_input = nn.Dropout1d(dropout)

        self.conv1 = nn.Sequential(
            nn.Conv1d(self.vocab_size, feature_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(feature_size, feature_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(feature_size, feature_size, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)  # output shape (batch, features, 1)
        )

        # Calculate output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, self.vocab_size, self.max_word_length)
            dummy_out = self._forward_conv(dummy)
            self.flatten_dim = dummy_out.view(1, -1).shape[1]

        self.fc = nn.Linear(self.flatten_dim, emb_size)

    def _forward_conv(self, x):
        x = self.dropout_input(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x) 
        x = x.view(x.size(0), -1) 
        return self.fc(x)
    
    def _word_to_ohe(self, word): # make oh representation of a word and pad to max_word_len
        idxs = [self.char_to_idx.get(c, self.unk_idx) for c in word[:self.max_word_length]] #get oh encoding for each char in word
        one_hot = torch.eye(self.vocab_size)[idxs].T  # make 0/1 matrix with shape (vocab_size, len)
        padded = nn.functional.pad(one_hot, (0, self.max_word_length - one_hot.shape[1]))  # (vocab_size, max_word_length)
        return padded
   
    def batch_cnn_embedding_generator(self, text, max_sent_len, batch_size):                 
        for i in range(0, len(text), batch_size):
            batch_sentences = text[i:i + batch_size]

            word_tensors = []
            for sentence in batch_sentences:
                sent_tensors = []
                for word in sentence[:max_sent_len]:
                    word = word.lower() 
                    ohe = self._word_to_ohe(word).unsqueeze(0)
                    sent_tensors.append(ohe)
                while len(sent_tensors) < max_sent_len:
                    sent_tensors.append(torch.zeros((1, self.vocab_size, self.max_word_length)))
                word_tensors.append(torch.cat(sent_tensors, dim=0))  

            batch_tensor = torch.stack(word_tensors) 
            batch_tensor = batch_tensor.view(-1, self.vocab_size, self.max_word_length)  
            embeddings = self.forward(batch_tensor)  
            embeddings = embeddings.view(len(batch_sentences), max_sent_len, -1) 
            yield embeddings