# python fine_tune.py --task twitter_semeval --test-size 0.2 --model bert-base-uncased --epochs 20 --learning-rate 1e-5 --train-bs 8 --warmup-steps 50 --logging-steps 50
# python fine_tune.py --task twitter_semeval --test-size 0.2 --model bert-base-cased --epochs 20 --learning-rate 1e-5 --train-bs 8 --warmup-steps 50 --logging-steps 50
# python fine_tune.py --task twitter_semeval --test-size 0.2 --model roberta-base --epochs 20 --learning-rate 1e-5 --train-bs 8 --warmup-steps 50 --logging-steps 50
# python fine_tune.py --task twitter_semeval --test-size 0.2 --model xlnet-base-cased --epochs 20 --learning-rate 1e-5 --train-bs 8 --warmup-steps 50 --logging-steps 50

python fine_tune.py --task twitter_semeval --test-size 0.2 --model albert-base-v2 --epochs 20 --learning-rate 1e-5 --train-bs 8 --warmup-steps 50 --logging-steps 50
python fine_tune.py --task twitter_semeval --test-size 0.2 --model microsoft/mpnet-base --epochs 20 --learning-rate 1e-5 --train-bs 8 --warmup-steps 50 --logging-steps 50
python fine_tune.py --task twitter_semeval --test-size 0.2 --model microsoft/deberta-base --epochs 20 --learning-rate 1e-5 --train-bs 8 --warmup-steps 50 --logging-steps 50
python fine_tune.py --task twitter_semeval --test-size 0.2 --model facebook/muppet-roberta-base --epochs 20 --learning-rate 1e-5 --train-bs 8 --warmup-steps 50 --logging-steps 50
python fine_tune.py --task twitter_semeval --test-size 0.2 --model google/electra-base-generator --epochs 20 --learning-rate 1e-5 --train-bs 8 --warmup-steps 50 --logging-steps 50
