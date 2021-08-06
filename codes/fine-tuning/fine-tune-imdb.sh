# python fine_tune.py --task imdb --model bert-base-uncased --train-bs 8 --learning-rate 1e-5 --epochs 10 --warmup-steps 50 --logging-steps 50
# python fine_tune.py --task imdb --model bert-base-cased --train-bs 8 --learning-rate 1e-5 --epochs 10 --warmup-steps 50 --logging-steps 50
# python fine_tune.py --task imdb --model roberta-base --train-bs 8 --learning-rate 1e-5 --epochs 10 --warmup-steps 50 --logging-steps 50
# python fine_tune.py --task imdb --model xlnet-base-cased --train-bs 8 --learning-rate 1e-5 --epochs 10 --warmup-steps 50 --logging-steps 50

python fine_tune.py --task imdb --model albert-base-v2 --train-bs 8 --learning-rate 1e-5 --epochs 10 --warmup-steps 50 --logging-steps 50
python fine_tune.py --task imdb --model microsoft/mpnet-base --train-bs 8 --learning-rate 1e-5 --epochs 10 --warmup-steps 50 --logging-steps 50
python fine_tune.py --task imdb --model microsoft/deberta-base --train-bs 4 --learning-rate 1e-5 --epochs 10 --warmup-steps 50 --logging-steps 50
python fine_tune.py --task imdb --model google/electra-base-generator --train-bs 8 --learning-rate 1e-5 --epochs 10 --warmup-steps 50 --logging-steps 50
python fine_tune.py --task imdb --model facebook/muppet-roberta-base --train-bs 8 --learning-rate 1e-5 --epochs 10 --warmup-steps 50 --logging-steps 50

