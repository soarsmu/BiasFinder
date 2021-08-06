## genereate mutants using biasfinder on Twitter S140 dataset
## A model which is fine-tuned using Twitter S140 dataset is used for predicting the mutants
python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model bert-base-uncased --mutant twitter_s140
python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model bert-base-cased --mutant twitter_s140
python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model roberta-base --mutant twitter_s140

python predict.py --mutation-tool eec --bias-type gender --task twitter_s140 --model bert-base-uncased --mutant twitter_s140
python predict.py --mutation-tool eec --bias-type gender --task twitter_s140 --model bert-base-cased --mutant twitter_s140
python predict.py --mutation-tool eec --bias-type gender --task twitter_s140 --model roberta-base --mutant twitter_s140
