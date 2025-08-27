import itertools
import seqeval.metrics
import seqeval.scheme
import torch
import numpy as np
import matplotlib.pyplot as plt
import seqeval 

class Evaluation:
    def __init__(self, emb_model, tagging_scheme = 'IOB1'):
        if tagging_scheme == 'IOB1':
            self.tagging_scheme = seqeval.scheme.IOB1
        elif tagging_scheme == 'IOBES':
            self.tagging_scheme = seqeval.scheme.IOBES
        else:
            raise ValueError(f"Tagging scheme {tagging_scheme} not supported")
        self.emb_model = emb_model

    def _get_preds(self, data_loader, char_embeddings, device, ft_bb, model, num_to_tag_dict):
        all_true_tags = []
        all_pred_tags = []
        final_loss = 0
        for (tokens, tags, emb_att_mask, crf_mask), char_embedding in zip(data_loader, char_embeddings or itertools.repeat(None)): 
                if char_embedding != None:
                        batch_char_embedding = char_embedding.to(device)
                else:
                    batch_char_embedding = None

                batch_attention_masks = emb_att_mask.to(device)
                batch_tags = tags.to(device)

                if ft_bb: #if doing bioBERT finetuning, model takes tokens
                    batch_tokens = tokens.to(device)
                    loss = model(batch_tokens, batch_tags, batch_attention_masks, batch_char_embedding)
                    final_loss += loss.item()
                    pred_tags = model.predict(batch_tokens, batch_attention_masks, batch_char_embedding) 

                else: #else, model takes emeddings
                    batch_embeddings = self.emb_model.get_embedding(tokens, emb_att_mask)
                    batch_embeddings = batch_embeddings.to(device)
                    loss = model(batch_embeddings, batch_tags, batch_attention_masks, batch_char_embedding)
                    final_loss += loss.item()
                    pred_tags = model.predict(batch_embeddings, batch_attention_masks, batch_char_embedding) 


                #remove padding
                relevant_true_tags = self.emb_model.get_relevant_tags(tags, num_to_tag_dict, crf_mask)

                # Predicted tags already have no padding, so just map them to strings
                relevant_pred_tags = self.emb_model.get_relevant_tags(pred_tags, num_to_tag_dict, crf_mask) 
                
                all_true_tags.extend(relevant_true_tags)  
                all_pred_tags.extend(relevant_pred_tags)
        return all_true_tags, all_pred_tags, final_loss
        
    def evaluate(self, data_loader, model, device, char_embeddings, num_to_tag_dict, logger, ft_bb=False, epoch = -1):
        model = model.to(device)    
        model.eval()  # Set model to evaluation mode
        all_true_tags = []
        all_pred_tags = []
        with torch.no_grad():
            all_true_tags, all_pred_tags, final_loss = self._get_preds(data_loader, char_embeddings, device, ft_bb, model, num_to_tag_dict)

            #mode='strict' ->  ensures that entity predictions are only counted as correct if they exactly match the true entity boundaries and the entity type
            f1_score = seqeval.metrics.f1_score(all_true_tags, all_pred_tags, average='micro') 
            f1_score_strict = seqeval.metrics.f1_score(all_true_tags, all_pred_tags, average='micro', mode='strict', scheme=self.tagging_scheme)
            loss = final_loss/len(data_loader)

            if epoch == -1: #test set
                precision = seqeval.metrics.precision_score(all_true_tags, all_pred_tags, average='micro') 
                precision_strict = seqeval.metrics.precision_score(all_true_tags, all_pred_tags, average='micro', mode='strict', scheme=self.tagging_scheme)
                recall = seqeval.metrics.recall_score(all_true_tags, all_pred_tags, average='micro') 
                recall_strict = seqeval.metrics.recall_score(all_true_tags, all_pred_tags, average='micro', mode='strict', scheme=self.tagging_scheme)
                logger.log_test_results(loss, f1_score, precision, recall, f1_score_strict, precision_strict, recall_strict)

                print(seqeval.metrics.classification_report(all_true_tags, all_pred_tags, digits=4), flush=True)
                print("strict: ")
                print(seqeval.metrics.classification_report(all_true_tags, all_pred_tags, digits=4,  mode='strict', scheme=self.tagging_scheme))
                print(f"Test Loss = {loss}")
            
            else:
                logger.log_val_results(epoch, loss, f1_score, f1_score_strict)
            
            return loss, f1_score_strict
        
    def hyperparam_eval(self, data_loader, model, device, char_embeddings, num_to_tag_dict, ft_bb=False):        
        model = model.to(device)    
        model.eval()  # Set model to evaluation mode
        all_true_tags = []
        all_pred_tags = []
        with torch.no_grad():
            all_true_tags, all_pred_tags, final_loss = self._get_preds(data_loader, char_embeddings, device, ft_bb, model, num_to_tag_dict)

            #mode='strict' ->  ensures that entity predictions are only counted as correct if they exactly match the true entity boundaries and the entity type
            f1_score_strict = seqeval.metrics.f1_score(all_true_tags, all_pred_tags, average='micro', mode='strict', scheme=self.tagging_scheme)
            loss = final_loss/len(data_loader)    

            return loss, f1_score_strict





        
        