declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base")

for model in ${models[@]}; do
    python fine_tune.py \
        --task imdb \
        --model $model \
        --train-bs 8 \
        --learning-rate 1e-5 \
        --epochs 20 \
        --warmup-steps 50 \
        --logging-steps 50
done
