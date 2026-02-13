import os
import torch
import torch.nn as nn
from TorchCRF import CRF
from transformers import AutoTokenizer, AutoModel
import settings

class BiRNN_CRF(nn.Module):
    def __init__(self, num_tag, model_args, word_embedding_dim, char_embedding_dim = None): 
        super(BiRNN_CRF, self).__init__()
        self.num_tag = num_tag

        self.cell = model_args['cell']
        self.hidden_size = model_args['hidden_size']
        self.num_layers = model_args['num_layers']   
        self.dropout = model_args['dropout']
        self.use_crf = model_args['use_crf']
        self.attention = model_args['attention']
        
        self.embedding_dim = word_embedding_dim + char_embedding_dim if char_embedding_dim != None else word_embedding_dim

        if self.cell == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        elif self.cell == 'gru':    
            self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        else:       
            raise ValueError(f"Cell {self.cell} not supported")

        self.normalize = nn.LayerNorm(self.hidden_size*2) 
        self.dropout_tag = nn.Dropout(self.dropout)

        if self.attention:
            self.attention_layer = nn.MultiheadAttention(self.hidden_size*2, model_args['att_num_of_heads'], batch_first=True) 
            self.hidden2tag_tag = nn.Linear(self.hidden_size*4, self.num_tag)
        else:           
            self.hidden2tag_tag = nn.Linear(self.hidden_size*2, self.num_tag) 

        if self.use_crf:
            self.crf_tag = CRF(self.num_tag)
        else:
            if model_args['loss'] == 'cross_entropy':
                self.criterion = nn.CrossEntropyLoss(ignore_index=-1) 
            else:
                raise ValueError(f"Loss {model_args['loss']} not supported")

    
    # Return the loss only, does not decode tags
    def forward(self, word_embedding, target_tag, attention_mask, char_embedding = None): 
        mask = attention_mask.bool()
        if char_embedding != None:
            embedding = torch.cat((word_embedding, char_embedding), dim=-1)
        else:
            embedding = word_embedding
        h, _ = self.rnn(embedding)
        h_norm = self.normalize(h)
        o_tag = self.dropout_tag(h_norm)

        if self.attention:
            padding_mask = torch.where(mask == True, False, True) #key_padding_mask expects True on indexes that should be ignored
            attention_output, _ = self.attention_layer(o_tag, o_tag, o_tag, key_padding_mask = padding_mask, need_weights=False)  
            tag = self.hidden2tag_tag(torch.cat([attention_output, o_tag], dim=-1))
        else:
            tag = self.hidden2tag_tag(o_tag)

        if self.use_crf:
            loss = -self.crf_tag.forward(tag, target_tag, mask).mean()
        else:  
            loss = self.criterion(tag.view(-1, self.num_tag), target_tag.view(-1))

        return loss

    def predict(self, word_embedding, attention_mask, char_embedding = None): 
        mask = attention_mask.bool()
        if char_embedding != None:
            embedding = torch.cat((word_embedding, char_embedding), dim=-1) 
        else:
            embedding = word_embedding
        h, _ = self.rnn(embedding)
        h_norm = self.normalize(h)
        o_tag = self.dropout_tag(h_norm)

        if self.attention:
            padding_mask = torch.where(mask == True, False, True) #key_padding_mask expects True on indexes that should be ignored
            attention_output, _ = self.attention_layer(o_tag, o_tag, o_tag, key_padding_mask = padding_mask, need_weights=False)  
            tag = self.hidden2tag_tag(torch.cat([attention_output, o_tag], dim=-1))
        else:
            tag = self.hidden2tag_tag(o_tag)
            
        if self.use_crf:
            tags = self.crf_tag.viterbi_decode(tag, mask)
            predicted_tags = [[torch.tensor(t) for t in tag] for tag in tags]
        else:
            predicted_tags = torch.argmax(tag, dim=-1)

        return predicted_tags
    
class ft_bb_BiRNN_CRF(nn.Module):
    def __init__(self, num_tag, model_args, char_embedding_dim = None): 
        super(ft_bb_BiRNN_CRF, self).__init__()
        self.num_tag = num_tag

        self.cell = model_args['cell']
        self.hidden_size = model_args['hidden_size']
        self.num_layers = model_args['num_layers']   
        self.dropout = model_args['dropout']
        self.use_crf = model_args['use_crf']
        self.attention = model_args['attention']

        bioBERT_setup_path = os.path.join(settings.EMBEDDINGS_PATH, "bioBERT_setup") 
        self.tokenizer = AutoTokenizer.from_pretrained(bioBERT_setup_path)
        self.bert = AutoModel.from_pretrained(bioBERT_setup_path)
            
        self.embedding_dim = self.bert.config.hidden_size + char_embedding_dim if char_embedding_dim != None else self.bert.config.hidden_size

        if self.cell == 'lstm':
            self.rnn = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        elif self.cell == 'gru':    
            self.rnn = nn.GRU(self.embedding_dim, self.hidden_size, num_layers=self.num_layers, bidirectional=True, batch_first=True)
        else:       
            raise ValueError(f"Cell {self.cell} not supported")

        self.normalize = nn.LayerNorm(self.hidden_size*2)
        self.dropout_tag = nn.Dropout(self.dropout)

        if self.attention:
            self.attention_layer = nn.MultiheadAttention(self.hidden_size*2, model_args['att_num_of_heads'], batch_first=True) 
            self.hidden2tag_tag = nn.Linear(self.hidden_size*4, self.num_tag) 
        else:
            self.hidden2tag_tag = nn.Linear(self.hidden_size*2, self.num_tag) 
        if self.use_crf:
            self.crf_tag = CRF(self.num_tag)
        else:
            if model_args['loss'] == 'cross_entropy':
                self.criterion = nn.CrossEntropyLoss(ignore_index=-1) 
            else:
                raise ValueError(f"Loss {model_args['loss']} not supported")

    
    # Return the loss only, does not decode tags
    def forward(self, tokens, target_tag, attention_mask, char_embedding = None): 
        bert_out = self.bert(input_ids=tokens, attention_mask=attention_mask)
        bert_embeddings = bert_out.last_hidden_state
        mask = attention_mask.bool()
        if char_embedding != None:
            embedding = torch.cat((bert_embeddings, char_embedding), dim=-1) 
        else:
            embedding = bert_embeddings
        h, _ = self.rnn(embedding)
        h_norm = self.normalize(h)
        o_tag = self.dropout_tag(h_norm)

        if self.attention:
            padding_mask = torch.where(mask == True, False, True) #key_padding_mask expects True on indexes that should be ignored
            attention_output, _ = self.attention_layer(o_tag, o_tag, o_tag, key_padding_mask = padding_mask, need_weights=False)  
            tag = self.hidden2tag_tag(torch.cat([attention_output, o_tag], dim=-1))
        else:
            tag = self.hidden2tag_tag(o_tag)

        if self.use_crf:
            loss = -self.crf_tag.forward(tag, target_tag, mask).mean()
        else:  
            loss = self.criterion(tag.view(-1, self.num_tag), target_tag.view(-1))

        return loss

    def predict(self, tokens, attention_mask, char_embedding = None): 
        bert_out = self.bert(input_ids=tokens, attention_mask=attention_mask)
        bert_embeddings = bert_out.last_hidden_state
        mask = attention_mask.bool()
        if char_embedding != None:
            embedding = torch.cat((bert_embeddings, char_embedding), dim=-1) 
        else:
            embedding = bert_embeddings
        h, _ = self.rnn(embedding)
        h_norm = self.normalize(h)
        o_tag = self.dropout_tag(h_norm)

        if self.attention:
            padding_mask = torch.where(mask == True, False, True) #key_padding_mask expects True on indexes that should be ignored
            attention_output, _ = self.attention_layer(o_tag, o_tag, o_tag, key_padding_mask = padding_mask, need_weights=False)  
            tag = self.hidden2tag_tag(torch.cat([attention_output, o_tag], dim=-1))
        else:
            tag = self.hidden2tag_tag(o_tag)
            
        if self.use_crf:
            tags = self.crf_tag.viterbi_decode(tag, mask)
            predicted_tags = [[torch.tensor(t) for t in tag] for tag in tags]
        else:
            predicted_tags = torch.argmax(tag, dim=-1)

        return predicted_tags
    


