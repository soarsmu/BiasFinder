## genereate mutants using biasfinder on Twitter SemEval dataset
## A model which is fine-tuned using Twitter SemEval dataset is used for predicting the mutants
# python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_semeval --model bert-base-uncased --mutant twitter_semeval
# python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_semeval --model bert-base-cased --mutant twitter_semeval
# python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_semeval --model roberta-base --mutant twitter_semeval
python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_semeval --model albert-base-v2 --mutant twitter_semeval
python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_semeval --model microsoft/mpnet-base --mutant twitter_semeval
python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_semeval --model microsoft/deberta-base --mutant twitter_semeval
python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_semeval --model facebook/muppet-roberta-base --mutant twitter_semeval
python predict.py --mutation-tool biasfinder --bias-type gender --task twitter_semeval --model google/electra-base-generator --mutant twitter_semeval

# python predict.py --mutation-tool eec --bias-type gender --task twitter_semeval --model bert-base-uncased --mutant twitter_semeval
# python predict.py --mutation-tool eec --bias-type gender --task twitter_semeval --model bert-base-cased --mutant twitter_semeval
# python predict.py --mutation-tool eec --bias-type gender --task twitter_semeval --model roberta-base --mutant twitter_semeval
python predict.py --mutation-tool eec --bias-type gender --task twitter_semeval --model albert-base-v2 --mutant twitter_semeval
python predict.py --mutation-tool eec --bias-type gender --task twitter_semeval --model microsoft/mpnet-base --mutant twitter_semeval
python predict.py --mutation-tool eec --bias-type gender --task twitter_semeval --model microsoft/deberta-base --mutant twitter_semeval
python predict.py --mutation-tool eec --bias-type gender --task twitter_semeval --model facebook/muppet-roberta-base --mutant twitter_semeval
python predict.py --mutation-tool eec --bias-type gender --task twitter_semeval --model google/electra-base-generator --mutant twitter_semeval
