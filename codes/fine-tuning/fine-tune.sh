# python fine_tune.py --model microsoft/deberta-base --gpu-id gpu0 --train-bs 4
# python fine_tune.py --model roberta-base --gpu-id gpu0 --train-bs 8
# python fine_tune.py --model albert-base-v1 --gpu-id gpu0 --train-bs 8
# python fine_tune.py --model albert-base-v2 --gpu-id gpu0 --train-bs 8


python fine_tune.py --model microsoft/mpnet-base --gpu-id gpu1 --train-bs 8
python fine_tune.py --model google/electra-base-generator --gpu-id gpu1 --train-bs 8
python fine_tune.py --model facebook/muppet-roberta-base --gpu-id gpu1 --train-bs 8
python fine_tune.py --model xlnet-base-cased --gpu-id gpu1 --train-bs 8