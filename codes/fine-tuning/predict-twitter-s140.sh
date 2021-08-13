## genereate mutants using biasfinder on Twitter S140 dataset
## A model which is fine-tuned using Twitter S140 dataset is used for predicting the mutants
# python predict.py --type mutant --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model bert-base-uncased --mutant twitter_s140
# python predict.py --type mutant --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model bert-base-cased --mutant twitter_s140
# python predict.py --type mutant --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model roberta-base --mutant twitter_s140
# python predict.py --type mutant --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model albert-base-v2 --mutant twitter_s140
python predict.py --type mutant --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model xlnet-base-cased --mutant twitter_s140
# python predict.py --type mutant --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model microsoft/mpnet-base --mutant twitter_s140
# python predict.py --type mutant --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model microsoft/deberta-base --mutant twitter_s140
# python predict.py --type mutant --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model facebook/muppet-roberta-base --mutant twitter_s140
# python predict.py --type mutant --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model google/electra-base-generator --mutant twitter_s140

# python predict.py --type original --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model bert-base-uncased --mutant twitter_s140
# python predict.py --type original --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model bert-base-cased --mutant twitter_s140
# python predict.py --type original --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model roberta-base --mutant twitter_s140
# python predict.py --type original --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model albert-base-v2 --mutant twitter_s140
python predict.py --type original --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model xlnet-base-cased --mutant twitter_s140
# python predict.py --type original --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model microsoft/mpnet-base --mutant twitter_s140
# python predict.py --type original --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model microsoft/deberta-base --mutant twitter_s140
# python predict.py --type original --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model facebook/muppet-roberta-base --mutant twitter_s140
# python predict.py --type original --mutation-tool biasfinder --bias-type gender --task twitter_s140 --model google/electra-base-generator --mutant twitter_s140


# python predict.py --mutation-tool eec --bias-type gender --task twitter_s140 --model bert-base-uncased --mutant twitter_s140
# python predict.py --mutation-tool eec --bias-type gender --task twitter_s140 --model bert-base-cased --mutant twitter_s140
# python predict.py --mutation-tool eec --bias-type gender --task twitter_s140 --model roberta-base --mutant twitter_s140
# python predict.py --mutation-tool eec --bias-type gender --task twitter_s140 --model albert-base-v2 --mutant twitter_s140
# python predict.py --mutation-tool eec --bias-type gender --task twitter_s140 --model microsoft/mpnet-base --mutant twitter_s140
# python predict.py --mutation-tool eec --bias-type gender --task twitter_s140 --model microsoft/deberta-base --mutant twitter_s140
# python predict.py --mutation-tool eec --bias-type gender --task twitter_s140 --model facebook/muppet-roberta-base --mutant twitter_s140
# python predict.py --mutation-tool eec --bias-type gender --task twitter_s140 --model google/electra-base-generator --mutant twitter_s140
