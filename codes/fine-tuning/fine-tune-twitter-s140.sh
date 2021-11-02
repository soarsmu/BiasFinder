declare -a models=("bert-base-uncased" "bert-base-cased" "roberta-base" "xlnet-base-cased" "albert-base-v2" "microsoft/mpnet-base" "google/electra-base-generator" "facebook/muppet-roberta-base" "microsoft/deberta-base")

for model in ${models[@]}; do
    python fine_tune.py \
        --task twitter_s140 \
        --test-size 0.02 \
        --model $model \
        --epochs 30 \
        --learning-rate 1e-5 \
        --train-bs 32 \
        --warmup-steps 200 \
        --logging-steps 200 \
        --eval-steps 200 \
        --save-steps 200 
done