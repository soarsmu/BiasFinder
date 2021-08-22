## genereate mutants using biasfinder on Twitter S140 dataset
## A model which is fine-tuned using Twitter S140 dataset is used for predicting the mutants

# declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")
# # declare -a models=("bert-base-uncased" "bert-base-cased" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator")

# for model in ${models[@]}; do
#     python predict.py \
#         --type mutant \
#         --mutation-tool biasfinder \
#         --bias-type gender \
#         --task twitter_s140 \
#         --model bert-base-uncased \
#         --mutant twitter_s140
# done

# declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")
# for model in ${models[@]}; do
#     python predict.py \
#         --type original \
#         --mutation-tool biasfinder \
#         --bias-type gender \
#         --task twitter_s140 \
#         --model bert-base-uncased \
#         --mutant twitter_s140
# done

# declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")
# for model in ${models[@]}; do
#     python predict.py \
#         --type mutant \
#         --mutation-tool eec \
#         --bias-type gender \
#         --task twitter_s140 \
#         --model bert-base-uncased \
#         --mutant twitter_s140
# done


# declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base" "microsoft/deberta-base")
declare -a models=("roberta-base" "facebook/muppet-roberta-base")

for model in ${models[@]}; do
    python predict.py \
        --type mutant \
        --mutation-tool biasfinder \
        --bias-type country \
        --task twitter_s140 \
        --model bert-base-uncased \
        --mutant twitter_s140
done

# # declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base" "microsoft/deberta-base")
declare -a models=("roberta-base" "facebook/muppet-roberta-base")
for model in ${models[@]}; do
    python predict.py \
        --type original \
        --mutation-tool biasfinder \
        --bias-type country \
        --task twitter_s140 \
        --model bert-base-uncased \
        --mutant twitter_s140
done


# # declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base" "microsoft/deberta-base")
# declare -a models=("bert-base-uncased" "bert-base-cased" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator")
declare -a models=("roberta-base" "facebook/muppet-roberta-base")

for model in ${models[@]}; do
    python predict.py \
        --type mutant \
        --mutation-tool biasfinder \
        --bias-type occupation \
        --task twitter_s140 \
        --model bert-base-uncased \
        --mutant twitter_s140
done

# # declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base" "microsoft/deberta-base")
declare -a models=("roberta-base" "facebook/muppet-roberta-base")

for model in ${models[@]}; do
    python predict.py \
        --type original \
        --mutation-tool biasfinder \
        --bias-type occupation \
        --task twitter_s140 \
        --model bert-base-uncased \
        --mutant twitter_s140
done
