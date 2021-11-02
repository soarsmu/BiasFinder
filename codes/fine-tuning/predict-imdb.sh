## Models which are fine-tuned using Twitter S140 dataset are used to predict the mutants

declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")
for model in ${models[@]}; do
    python predict.py \
        --type mutant \
        --mutation-tool biasfinder \
        --bias-type gender \
        --task imdb \
        --model $model \
        --mutant imdb
done

declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")
for model in ${models[@]}; do
    python predict.py \
        --type original \
        --mutation-tool biasfinder \
        --bias-type gender \
        --task imdb \
        --model $model \
        --mutant imdb
done

declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")
for model in ${models[@]}; do
    python predict.py \
        --type mutant \
        --mutation-tool eec \
        --bias-type gender \
        --task imdb \
        --model $model \
        --mutant imdb
done

declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")
for model in ${models[@]}; do
    python predict.py \
        --type original \
        --mutation-tool eec \
        --bias-type gender \
        --task imdb \
        --model $model \
        --mutant imdb
done


declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")

for model in ${models[@]}; do
    python predict.py \
        --type mutant \
        --mutation-tool mtnlp \
        --bias-type gender \
        --task imdb \
        --model $model \
        --mutant imdb
done

declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")

for model in ${models[@]}; do
    python predict.py \
        --type original \
        --mutation-tool mtnlp \
        --bias-type gender \
        --task imdb \
        --model $model \
        --mutant imdb
done

declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")
for model in ${models[@]}; do
    python predict.py \
        --type mutant \
        --mutation-tool biasfinder \
        --bias-type country \
        --task imdb \
        --model $model \
        --mutant imdb
done

declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")
for model in ${models[@]}; do
    python predict.py \
        --type original \
        --mutation-tool biasfinder \
        --bias-type country \
        --task imdb \
        --model $model \
        --mutant imdb
done


declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")
for model in ${models[@]}; do
    python predict.py \
        --type mutant \
        --mutation-tool biasfinder \
        --bias-type occupation \
        --task imdb \
        --model $model \
        --mutant imdb
done

declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")

for model in ${models[@]}; do
    python predict.py \
        --type original \
        --mutation-tool biasfinder \
        --bias-type occupation \
        --task imdb \
        --model $model \
        --mutant imdb
done
