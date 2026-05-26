import json


files = ["/home/martavidas/Documents/FER/Diplomski/Diplomski/data/BRONCO150/bronco150/trf/bronco150_ner_train.json", 
         "/home/martavidas/Documents/FER/Diplomski/Diplomski/data/ncbi/ncbi_disease_json/ncbi_trf/ncbi_ner_train.json", 
         "/home/martavidas/Documents/FER/Diplomski/Diplomski/data/distemist/distemist/trf/distemist_ner_train.json", 
         "/home/martavidas/Documents/FER/Diplomski/Diplomski/data/quaero/quaero/trf/quaero_ner_train.json",
         "/home/martavidas/Documents/FER/Diplomski/Diplomski/data/quaero/quaero/trf/emea/emea_train.json",
         "/home/martavidas/Documents/FER/Diplomski/Diplomski/data/quaero/quaero/trf/medline/medline_train.json"]

for input_file in files:
    with open(input_file, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

    entity = 0
    no_entity = 0
    for item in data:
        if len(item["tokens"]) < 3:
            continue
        if len(item["entities"] ) == 0:
            no_entity += 1
        else:
            entity += 1

    print(no_entity/(no_entity+entity))

#Bronco  0.524679029957204

#NCBI  0.4822883457314913

#DISTEMIST 0.7038125948406677

#QUAERO 0.5266237565827969 (EMEA 0.5874769797421732, MEDLINE 0.4611451942740286)
