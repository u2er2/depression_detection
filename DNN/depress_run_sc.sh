python3 train.py -d ../data/depress_data/depress_train.csv --trial ../data/depress_data/depress_test.csv --iterations 10 --word_list ../data/word_list/depress.txt --emb ../data/glove.840B.300d.txt -o output_dir -b 512 --epochs 15 --lr 0.001 --maxlen 100 -t HHMM_transformer --task 1 --ablate_cat_emb
