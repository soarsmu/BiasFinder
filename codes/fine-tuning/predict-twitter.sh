## genereate mutants using biasfinder on Twitter S140 dataset
## A model which is fine-tuned using Twitter SemEval dataset is used for predicting the mutants
# python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_semeval --model bert-base-uncased --mutant twitter_s140

# python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_semeval --model bert-base-uncased --mutant twitter_semeval

python predict.py --mutation-tool eec --bias-type gender --task twitter_semeval --model bert-base-uncased --mutant twitter_semeval
