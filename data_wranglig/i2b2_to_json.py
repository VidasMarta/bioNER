import json
import os


def parse_text_file(file_path, path_to_save): 
    json_list = []
    json_format = dict()

    sentence,tag=[],[]

    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines: 
                line = line.strip()
                if len(line) == 0:
                    if sentence== [] and tag==[]:
                        continue
                    json_format["tags"] = tag
                    json_format["tokens"] = sentence
                    json_list.append(json_format)
                    json_format = dict()
                    sentence, tag = [], []
                else:
                    word, label = line.split(" ")
                    sentence.append(word)
                    if label == 'B-problem':
                        tag.append(0) #"B-Disease": 0
                    elif label == 'I-problem':
                        tag.append(1) #"I-Disease": 1
                    else:
                        tag.append(2) #"O": 2

        with open(path_to_save, "a") as f:
            for item in json_list:
                f.write(json.dumps(item) + "\n")

        print(f"Saved json to {path_to_save}")
    else:
        print(f"File {file_path} not found")


# Example usage
if __name__ == "__main__":
   i2b2_path = "/home/martavidas/Documents/FER/Diplomski/Diplomski/data/i2b2"
   train_init = os.path.join(i2b2_path, "train.txt")
   train_json = os.path.join(i2b2_path, "train_100pct.json")
   parse_text_file(train_init, train_json)


   dev_init = os.path.join(i2b2_path, "dev.txt")
   dev_json = os.path.join(i2b2_path, "devel.json")
   parse_text_file(dev_init, dev_json)


   test_init = os.path.join(i2b2_path, "test.txt")
   test_json = os.path.join(i2b2_path, "test.json")
   parse_text_file(test_init, test_json)