from abc import ABC, abstractmethod
import copy
import itertools

import torch
import models
from torch.optim import *
from torch.nn.utils import clip_grad_norm_
import settings

class Trainer(ABC):
    def __init__(self, model_name, model_args, num_tags, train_data_loader, valid_data_loader, word_embeddings_model, char_emb, text_train, text_val, 
                 max_len, batch_size, device, num_to_tag, eval, logger):
        self.model_name = model_name
        self.model_args = model_args
        self.max_grad_norm = float(model_args['max_grad_norm'])
        self.num_epochs = int(model_args['epochs'])
        self.num_tags = num_tags
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.word_embeddings_model = word_embeddings_model
        self.char_emb = char_emb
        self.text_train = text_train
        self.text_val = text_val
        self.max_len = max_len
        self.batch_size = batch_size
        self.device = device
        self.num_to_tag = num_to_tag
        self.eval = eval
        self.logger = logger
            

    @abstractmethod
    def _define_optimizer(self): 
        pass

    @abstractmethod
    def _train_one_epoch(self, data_loader, char_embeddings):
        pass
    
    def train(self, save_model=True): #set to True when not using hyperparameter tuning
        self.optimizer = self._define_optimizer()
        self.model.to(self.device)

        # Add LR scheduler
        if save_model:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=4,
                threshold=1e-3, 
                threshold_mode='rel',
                verbose=True
            )

        #Initialize Variables for EarlyStopping
        best_f1 = -1
        best_model_weights = None
        patience = int(self.model_args['early_stopping'])
        epochs_no_improve = 0

        for epoch in range(self.num_epochs):
            if self.char_emb is not None:
                train_char_embeddings = self.char_emb.batch_cnn_embedding_generator(self.text_train, self.max_len, self.batch_size)
                val_char_embeddings = self.char_emb.batch_cnn_embedding_generator(self.text_val, self.max_len, self.batch_size)
            else:
                train_char_embeddings = None
                val_char_embeddings = None

            train_loss = self._train_one_epoch(self.train_data_loader, train_char_embeddings)
            if save_model:
                self.logger.log_train_loss(epoch+1, train_loss)

            # Validation
            if save_model:
                val_loss, f1 = self.eval.evaluate(self.valid_data_loader, self.model, self.device, val_char_embeddings, self.num_to_tag, self.logger, self.finetuning, epoch+1)
            else:
                val_loss, f1 = self.eval.hyperparam_eval(self.valid_data_loader, self.model, self.device, val_char_embeddings, self.num_to_tag, self.finetuning)

            #Step the scheduler with validation metric (F1)
            if save_model:
                scheduler.step(f1)

            #Early stopping looking at f1-score (strict)
            if f1 > best_f1:
                best_f1 = f1
                epochs_no_improve = 0
                best_model_weights = copy.deepcopy(self.model.state_dict())  # Deep copy here      
                print(f"Validation f1 improved to {best_f1:.4f} in epoch {epoch+1}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break
            
            if epoch%10 == 0:
                print(f"Train Loss ({epoch+1}/{self.num_epochs}) = {train_loss}")
                print(f"Validation Loss ({epoch+1}/{self.num_epochs}) = {val_loss}")

        # Save the best model
        if save_model:
            self.best_model.load_state_dict(best_model_weights)
            torch.save(self.best_model.state_dict(), settings.MODEL_PATH +f"/{self.model_name}_best.bin")
        
        else:
            return best_f1

class Finetuning_Trainer(Trainer):
    def __init__(self, model_name, model_args, num_tags, train_data_loader, valid_data_loader, word_embeddings_model, char_emb, text_train, text_val, 
                 max_len, batch_size, device, num_to_tag, eval, logger):
        super().__init__(model_name, model_args, num_tags, train_data_loader, valid_data_loader, word_embeddings_model, char_emb, text_train, text_val,
                          max_len, batch_size, device, num_to_tag, eval, logger)   
        self.finetuning = True
        self.model = models.ft_bb_BiRNN_CRF(num_tags, model_args, model_args['char_embedding_dim'])
        self.best_model = models.ft_bb_BiRNN_CRF(num_tags, model_args, model_args['char_embedding_dim'])

    def _define_optimizer(self):
        if self.model_args['optimizer'] == 'adam':
            return Adam([
                    {"params": self.model.bert.parameters(), "lr": float(self.model_args['ft_lr'])},
                    {"params": self.model.rnn.parameters(), "lr": float(self.model_args['lr'])},
                    {"params": self.model.hidden2tag_tag.parameters(), "lr": float(self.model_args['lr'])},
                    {"params": self.model.crf_tag.parameters(), "lr": float(self.model_args['lr'])}
                    ])
        elif self.model_args['optimizer'] == 'adamw':
            return AdamW([
                    {"params": self.model.bert.parameters(), "lr": float(self.model_args['ft_lr'])},
                    {"params": self.model.rnn.parameters(), "lr": float(self.model_args['lr'])},
                    {"params": self.model.hidden2tag_tag.parameters(), "lr": float(self.model_args['lr'])},
                    {"params": self.model.crf_tag.parameters(), "lr": float(self.model_args['lr'])}
                    ])
        elif self.model_args['optimizer'] == 'sgd':
            return SGD([
                    {"params": self.model.bert.parameters(), "lr": float(self.model_args['ft_lr'])},
                    {"params": self.model.rnn.parameters(), "lr": float(self.model_args['lr'])},
                    {"params": self.model.hidden2tag_tag.parameters(), "lr": float(self.model_args['lr'])},
                    {"params": self.model.crf_tag.parameters(), "lr": float(self.model_args['lr'])}
                    ])
        else:
            raise ValueError(f"Optimizer {self.model_args['optimizer']} not supported")
        
    def _train_one_epoch(self, data_loader, char_embeddings):
        self.model.train()
        final_loss = 0
        for (tokens, tags, emb_att_mask, _), char_embedding in zip(data_loader, char_embeddings or itertools.repeat(None)): 
            self.optimizer.zero_grad()

            batch_tokens = tokens.to(self.device)
            batch_tags = tags.to(self.device)
            batch_attention_masks = emb_att_mask.to(self.device)

            if char_embedding != None:
                batch_char_embedding = char_embedding.to(self.device)
            else:
                batch_char_embedding = None

            loss = self.model(batch_tokens, batch_tags, batch_attention_masks, batch_char_embedding)
            loss.backward()
            if self.max_grad_norm:
                total_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print("Warning: gradient norm is NaN or Inf!")

            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

            
class Normal_Trainer(Trainer):
    def __init__(self, model_name, model_args, num_tags, train_data_loader, valid_data_loader, word_embeddings_model, char_emb, text_train, text_val, 
                 max_len, batch_size, device, num_to_tag, eval, logger):
        super().__init__(model_name, model_args, num_tags, train_data_loader, valid_data_loader, word_embeddings_model, char_emb, text_train, text_val,
                          max_len, batch_size, device, num_to_tag, eval, logger)
        
        self.finetuning = False
        self.model = models.BiRNN_CRF(num_tags, model_args, word_embeddings_model.embedding_dim, model_args['char_embedding_dim'])
        self.best_model = models.BiRNN_CRF(num_tags, model_args, word_embeddings_model.embedding_dim, model_args['char_embedding_dim'])

    def _define_optimizer(self):
        if self.model_args['optimizer'] == 'adam':
            return Adam(self.model.parameters(), lr=float(self.model_args['lr']))
        elif self.model_args['optimizer'] == 'adamw':
            return AdamW(self.model.parameters(), lr=float(self.model_args['lr']))
        elif self.model_args['optimizer'] == 'sgd':
            return SGD(self.model.parameters(), lr=float(self.model_args['lr'])) 
        else:
            raise ValueError(f"Optimizer {self.model_args['optimizer']} not supported")
        
    def _train_one_epoch(self, data_loader, char_embeddings):
        self.model.train()
        final_loss = 0
        for (tokens, tags, emb_att_mask, _), char_embedding in zip(data_loader, char_embeddings or itertools.repeat(None)): 
            self.optimizer.zero_grad()
            
            batch_embeddings = self.word_embeddings_model.get_embedding(tokens, emb_att_mask) 
            batch_embeddings = batch_embeddings.to(self.device)
            batch_tags = tags.to(self.device)
            batch_attention_masks = emb_att_mask.to(self.device)

            if char_embedding != None:
                batch_char_embedding = char_embedding.to(self.device)
            else:
                batch_char_embedding = None

            loss = self.model(batch_embeddings, batch_tags, batch_attention_masks, batch_char_embedding)
            loss.backward()
            if self.max_grad_norm:
                total_norm = clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                if torch.isnan(total_norm) or torch.isinf(total_norm):
                    print("Warning: gradient norm is NaN or Inf!")

            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)
