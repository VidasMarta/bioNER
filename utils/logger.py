import csv
import os
import json
from datetime import datetime
import numpy as np

class Logger:
    def __init__(self, model_logs_path, model_args=None, settings_args=None):
        self.output_path = model_logs_path
        self.init_time = datetime.now()
        if not os.path.exists(self.output_path):
            os.mkdir(model_logs_path)

        #prepare files for log writing
        open(self.output_path+'/train.log',"a").close()
        open(self.output_path+'/valid.log',"a").close()
        open(self.output_path+'/test.log',"a").close()


        #write config (hyperparameters etc.)
        log_config_file = open(self.output_path+'/config.log', "w")
        json.dump(model_args, log_config_file)
        log_config_file.write("\n")
        json.dump(settings_args, log_config_file)

        log_config_file.close()


    def log_test_results(self, loss, f1_score, precision, recall, f1_score_strict, precision_strict, recall_strict):
        log_file = open(self.output_path + '/test.log',"a")
        log_file.write(f"loss: {loss}, f1_score: {f1_score}, precison: {precision}, recall:{recall}, f1_score(strict): {f1_score_strict}, precision(strict): {precision_strict}, recall(strict): {recall_strict}, time: {datetime.now() - self.init_time}\n")
        log_file.close()

    def log_val_results(self, epoch, loss, f1_score, f1_score_strict):
        log_file = open(self.output_path + '/valid.log',"a")
        log_file.write(f"epoch: {epoch}, loss: {loss}, f1_score: {f1_score}, f1_score(strict): {f1_score_strict}, time: {datetime.now() - self.init_time}\n")
        log_file.close()

    def log_train_loss(self, epoch, loss):
        log_file = open(self.output_path + '/train.log', "a")
        log_file.write(f"epoch: {epoch}, loss: {loss}, time: {datetime.now() - self.init_time}\n")
        log_file.close()

    def calculate_mean_stddev(self, training_set, file_to_save):
        f1_scores = []
        precission_scores = []
        recall_scores = []
        f1_strict_scores = []

        with open(self.output_path + '/test.log', 'r') as file:
            for line in file:
                try:
                    f1_start = line.index("f1_score:") + len("f1_score:")
                    f1_end = line.index(",", f1_start)
                    f1 = float(line[f1_start:f1_end].strip())
                    f1_scores.append(f1)

                    p_start = line.index("precison:") + len("precison:")
                    p_end = line.index(",", p_start)
                    p = float(line[p_start:p_end].strip())
                    precission_scores.append(p)

                    r_start = line.index("recall:") + len("recall:")
                    r_end = line.index(",", r_start)
                    r = float(line[r_start:r_end].strip())
                    recall_scores.append(r)

                    f1s_start = line.index("f1_score(strict):") + len("f1_score(strict):")
                    f1s_end = line.index(",", f1s_start)
                    f1_strict = float(line[f1s_start:f1s_end].strip())
                    f1_strict_scores.append(f1_strict)
                except ValueError:
                    continue  # Skip lines that don't match
        
        f1_mean = np.mean(f1_scores)
        f1_std = np.std(f1_scores)
        f1_strict_mean = np.mean(f1_strict_scores)
        f1_strct_std = np.std(f1_strict_scores)

        p_mean = np.mean(precission_scores)
        p_std = np.std(precission_scores)

        r_mean = np.mean(recall_scores)
        r_std = np.std(recall_scores)

        new_data = {
            'Training set': training_set,
            'F1': f"{f1_mean*100:0.2f} +/- {f1_std*100:0.2f}",
            'Precision': f"{p_mean*100:0.2f} +/- {p_std*100:0.2f}",
            'Recall': f"{r_mean*100:0.2f} +/- {r_std*100:0.2f}"
        }

        file_exists = os.path.isfile(file_to_save)
        os.makedirs(os.path.dirname(file_to_save), exist_ok=True)

        
        with open(file_to_save, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=new_data.keys())
            
            if not file_exists:
                writer.writeheader()
                
            writer.writerow(new_data)

        mean_std = f"Mean f1_score: {f1_mean}, Std. dev. f1_score: {f1_std}, Mean f1_score (strict): {f1_strict_mean}, Std. dev. f1_score (strict): {f1_strct_std}\n Mean precision: {p_mean}, Std. dev precision: {p_std}, Mean recall: {r_mean}, Std. dev recall: {r_std}"

        log_file = open(self.output_path + '/test.log',"a")
        log_file.write(mean_std)
        log_file.close()

        print("---->", mean_std)

            

