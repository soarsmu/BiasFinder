## genereate mutants using biasfinder on IMDB test set
## A model which is fine-tuned using IMDB train set 
## This model is used for predicting the mutants

declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")

for model in ${models[@]}; do
    python predict.py \
            --batch-size 64 \
            --type mutant \
            --mutation-tool biasfinder \
            --bias-type country \
            --task imdb \
            --model $model \
            --mutant imdb
done

for model in ${models[@]}; do
    python predict.py \
            --batch-size 64 \
            --type original \
            --mutation-tool biasfinder \
            --bias-type country \
            --task imdb \
            --model $model \
            --mutant imdb
done

for model in ${models[@]}; do
    python predict.py \
            --batch-size 64 \
            --type mutant \
            --mutation-tool biasfinder \
            --bias-type country \
            --task twitter_s140 \
            --model $model \
            --mutant twitter_s140
done

for model in ${models[@]}; do
    python predict.py \
            --batch-size 64 \
            --type original \
            --mutation-tool biasfinder \
            --bias-type country \
            --task twitter_s140 \
            --model $model \
            --mutant twitter_s140
done


for model in ${models[@]}; do
    python predict.py \
            --batch-size 64 \
            --type mutant \
            --mutation-tool mtnlp \
            --bias-type country \
            --task imdb \
            --model $model \
            --mutant imdb
done

for model in ${models[@]}; do
    python predict.py \
            --batch-size 64 \
            --type original \
            --mutation-tool mtnlp \
            --bias-type country \
            --task imdb \
            --model $model \
            --mutant imdb
done

for model in ${models[@]}; do
    python predict.py \
            --batch-size 64 \
            --type mutant \
            --mutation-tool mtnlp \
            --bias-type country \
            --task twitter_s140 \
            --model $model \
            --mutant twitter_s140
done

for model in ${models[@]}; do
    python predict.py \
            --batch-size 64 \
            --type original \
            --mutation-tool mtnlp \
            --bias-type country \
            --task twitter_s140 \
            --model $model \
            --mutant twitter_s140
done

