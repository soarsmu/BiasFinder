## genereate mutants using biasfinder on IMDB test set
## A model which is fine-tuned using IMDB train set 
## This model is used for predicting the mutants
# python predict.py --mutation-tool biasfinder --bias-type gender --task imdb --model bert-base-uncased --mutant imdb
# python predict.py --mutation-tool biasfinder --bias-type gender --task imdb --model bert-base-cased --mutant imdb
# python predict.py --mutation-tool biasfinder --bias-type gender --task imdb --model roberta-base --mutant imdb
# python predict.py --mutation-tool biasfinder --bias-type gender --task imdb --model xlnet-base-cased --mutant imdb
# python predict.py --mutation-tool biasfinder --bias-type gender --task imdb --model albert-base-v1 --mutant imdb

python predict.py --mutation-tool biasfinder --bias-type gender --task imdb --model albert-base-v2 --mutant imdb
# python predict.py --mutation-tool biasfinder --bias-type gender --task imdb --model microsoft/mpnet-base --mutant imdb
python predict.py --mutation-tool biasfinder --bias-type gender --task imdb --model google/electra-base-generator --mutant imdb
python predict.py --mutation-tool biasfinder --bias-type gender --task imdb --model facebook/muppet-roberta-base --mutant imdb
python predict.py --mutation-tool biasfinder --bias-type gender --task imdb --model microsoft/deberta-base --mutant imdb
