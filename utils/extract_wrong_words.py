import itertools
import torch

from datasets import Dataset, DatasetLoader
import models
from preprocessing import CharEmbeddingCNN, Embedding
import settings

def _get_entity(words, gold_seq, num_to_tag, word_idx):
    gold_id = int(gold_seq[word_idx])
    if gold_id == -1:
        return words[word_idx]  # treat ignored subwords as single token
    
    gold_tag = num_to_tag[gold_id]
    if gold_tag == "O" or "-" not in gold_tag:
        return words[word_idx]  # no entity
    
    prefix, ent_type = gold_tag.split("-", 1)
    
    # walk backwards to find the beginning of the entity
    start = word_idx
    while start > 0:
        prev_id = int(gold_seq[start - 1])
        if prev_id == -1:  # skip ignored subwords
            start -= 1
            continue
        prev_tag = num_to_tag[prev_id]
        if not prev_tag.endswith(ent_type) or prev_tag.startswith("O"):
            break
        start -= 1
    
    # walk forwards to find the end of the entity
    end = word_idx
    while end + 1 < len(words):
        next_id = int(gold_seq[end + 1])
        if next_id == -1:  # skip ignored subwords
            end += 1
            continue
        next_tag = num_to_tag[next_id]
        if not next_tag.startswith("I-") or not next_tag.endswith(ent_type):
            break
        end += 1
    
    entity_tokens = words[start:end + 1]
    return " ".join(entity_tokens)

def _get_errors(text, data_loader, char_embeddings, device, model, num_to_tag):
    errors = []
    with torch.no_grad():
        for batch_idx, ((tokens, tags, emb_att_mask, crf_mask), char_embedding) in enumerate(zip(data_loader, char_embeddings or itertools.repeat(None))): 
            if char_embedding != None:
                    batch_char_embedding = char_embedding.to(device)
            else:
                batch_char_embedding = None

            batch_attention_masks = emb_att_mask.to(device)
            batch_tags = tags.to(device)

            batch_tokens = tokens.to(device)
            pred_tags = model.predict(batch_tokens, batch_attention_masks, batch_char_embedding) 

            for idx, (pred_seq, gold_seq, mask_seq) in enumerate(zip(pred_tags, batch_tags, crf_mask)):
                words = text[batch_idx * data_loader.batch_size + idx]  # original words
                word_ptr = 0

                for i, (pred, gold, m) in enumerate(zip(pred_seq, gold_seq, mask_seq)):
                    if m == 0:  # skip padding and subwords
                        continue

                    word = words[word_ptr]
                    word_ptr += 1

                    pred = pred.item()
                    gold = gold.item()

                    if pred != gold:
                        errors.append({
                            "idx": batch_idx * data_loader.batch_size + idx, 
                            "token": word,
                            "predicted": num_to_tag[pred],
                            "gold": num_to_tag[gold],
                            "gold entity": _get_entity(words, gold_seq, num_to_tag, word_ptr),
                        })

    return errors
    


def evaluate_and_find_errors(model_name, settings_args, model_args, device):
    dataset_loader = DatasetLoader(settings_args['dataset'], settings.DATA_PATH)

    batch_size=model_args['batch_size']
    max_len = model_args['max_length'] 
    word_embedding = settings_args['word_embedding']
    word_embeddings_model = Embedding.create(word_embedding, dataset_loader.dataset_name, max_len) 

    tag_to_num, (text_train, tags_train), (text_val, tags_val), (text_test, tags_test) = dataset_loader.load_data()
    tokens_val_padded, tags_val_padded, attention_masks_val, crf_mask_val = word_embeddings_model.tokenize_and_pad_text(text_val, tags_val)
    val_data = Dataset(tokens_val_padded, tags_val_padded, attention_masks_val, crf_mask_val)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size)

    tokens_test_padded, tags_test_padded, attention_masks_test, crf_mask_test= word_embeddings_model.tokenize_and_pad_text(text_test, tags_test)
    test_data = Dataset(tokens_test_padded, tags_test_padded, attention_masks_test, crf_mask_test)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size)

    num_to_tag = dict((v,k) for k,v in tag_to_num.items())
    num_tags = len(tag_to_num)

    model_weights = torch.load(settings.MODEL_PATH + f"/{model_name}_best.bin")
    model = models.ft_bb_BiRNN_CRF(num_tags, model_args, settings_args['cnn_embedding_dim']) 
    model.load_state_dict(model_weights)
    model.to(device)
    model.eval()

    if settings_args['char_cnn_embedding'] is not None:
        vocab = settings_args['cnn_vocab']
        char_emb_size = settings_args['cnn_embedding_dim']
        model_args['char_embedding_dim'] = char_emb_size
        feature_size = settings_args['feature_size']
        max_word_len = settings_args['cnn_max_word_len']
        char_emb = CharEmbeddingCNN(vocab, char_emb_size, feature_size, max_word_len)
        val_char_embeddings = char_emb.batch_cnn_embedding_generator(text_val, max_len, batch_size)
        test_char_embeddings = char_emb.batch_cnn_embedding_generator(text_test, max_len, batch_size)
    else:
        val_char_embeddings = None
        test_char_embeddings = None

    errors_val = _get_errors(text_val, val_data_loader, val_char_embeddings, device, model, num_to_tag)
    errors_test = _get_errors(text_test, test_data_loader, test_char_embeddings, device, model, num_to_tag)

    return errors_val, errors_test 
