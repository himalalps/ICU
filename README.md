# Iterative Contrastive Unlearning

![](./figure/framework.png)

## Specification of dependencies

The code has been verified on Python 3.8.19.

```bash
$ conda create -n icu python=3.8
$ conda activate icu
# Install the correct torch version depending on CUDA version from https://pytorch.org/
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch 
$ pip install -r requirements.txt
```

## Training Code

Training scripts can be found under `scripts` directory.

With one gpu ("cuda:0"), the command below starts training of GPT-NEO-125m.
```bash
# train-125m.sh
$ python ./train_neo_meanwhile_update.py \
--exp "exp0" --model_name "EleutherAI/gpt-neo-125m" \
--tokenizer_name "EleutherAI/gpt-neo-125m" \
--gpt2_name "openai-community/gpt2" \
--bert_name "google-bert/bert-base-uncased" \
--prefix_length 200 --suffix_length 200 --target_length 200 \
--device "cuda:0" --batch_size 8 --num_workers 8 --lr 5e-6 \
--uw 1.0 --lw 0.5 --kl 1.0 --f1 0.3 --bleu 0.01 --acc 0.5994 \
--el 0.0499 --dir "result/test"
```

With three gpus (0,1,2), the command below starts training of GPT-NEO-1.3B.
```bash
# train-1_3b.sh
deepspeed --include localhost:0,1,2 \
./train_neo_meanwhile_update_deepspeed.py \
--deepspeed_config ./config/deepspeed3.json \
--exp "exp0" --model_name "EleutherAI/gpt-neo-1.3B" \
--tokenizer_name "EleutherAI/gpt-neo-1.3B" \
--gpt2_name "openai-community/gpt2" \
--bert_name "google-bert/bert-base-uncased" \
--prefix_length 200 --suffix_length 200 --target_length 200 \
--batch_size 4 --num_workers 8 --lr 5e-6 \
--uw 1.0 --lw 0.5 --kl 1.0 --f1 0.3 --bleu 0.01 --acc 0.5994 \
--el 0.0499 --dir "result/test"
```

With six gpus (0,1,2,3,4,5), the command below starts training of GPT-NEO-1.3B.
```bash
# train-2_7b.sh
$ deepspeed --include localhost:0,1,2,3,4,5 \
./train_neo_meanwhile_update_deepspeed.py \
--deepspeed_config ./config/deepspeed6.json \
--exp "exp0" --model_name "EleutherAI/gpt-neo-2.7B" \
--tokenizer_name "EleutherAI/gpt-neo-2.7B" \
--gpt2_name "openai-community/gpt2" \
--bert_name "google-bert/bert-base-uncased" \
--prefix_length 200 --suffix_length 200 --target_length 200 \
--batch_size 4 --num_workers 8 --lr 5e-6 \
--uw 1.0 --lw 0.5 --kl 1.0 --f1 0.3 --bleu 0.01 --acc 0.5994 \
--el 0.0499 --dir "result/test1"
```

## Evaluation Code

### Downstream Tasks

You can test the code on the downstream tasks using the command below.
```bash
# valid.sh
$ python ./valid.py \
--model_name "./result/test/EleutherAI/gpt-neo-125m_exp0_lr5e-06_uw1.0_lw0.5_kl1.0_epoch19_updateboth" \
--tokenizer_name "EleutherAI/gpt-neo-125m" \
--prefix_length 512 --suffix_length 512 --device "cuda:0" \
--batch_size 32 --num_workers 48 \
--dir "./result/test" --cache "./.cache"
```

### Evaluating Unlearning

The original model can be evaluated using the command below.
```bash
# eval.sh
python ./eval.py --exp "all" \
--model_name "EleutherAI/gpt-neo-125m" \
--tokenizer_name "EleutherAI/gpt-neo-125m" \
--gpt2_name "openai-community/gpt2" \
--bert_name "google-bert/bert-base-uncased" \
--prefix_length 200 --suffix_length 200 --target_length 200 \
--device "cuda:0" --batch_size 8 --num_workers 8 \
--dir "./result/test"
```

### GPT

The related code is in `evaluation` directory. `test.ipynb` is more convenient than `api.py`.

1. Fill in your GPT-4 api key in the code.
2. Use `convert.py` to convert the results of previous files in Evaluating. (Rearranging the files according to the code may be necessary.)
3. Run the code inside `evaluation`.

The files should be rearranged to a tree following below structure:
```
evaluation
│   125mneo.csv     # the results of gpt-neo-125m on all
│   125mopt.csv     # the results of opt-125m on all
│   api.py
│   convert.py
│   lm_extraction_128_0.csv
│   prompt.py
│   test.ipynb
│
├───125m-0
│       and.csv         # the results of KUMPR on 0
│       neo.csv         # generated
|       opt.csv         # generated
│       ours.csv        # our results
│       results.json    # generated
```

## Data Preparation

### Datasets

The target data can be downloaded from [this link](https://github.com/ethz-spylab/lm-extraction-benchmark-data/tree/main/datasets).

Below are the validation datasets used and can be downloaded from open source.

- [ai2_arc](https://allenai.org/data/arc)

- [hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)

- [math_qa](https://huggingface.co/datasets/math_qa)

- [piqa](https://huggingface.co/datasets/ybisk/piqa)

- [super_glue](https://huggingface.co/datasets/super_glue)

- [winogrande](https://huggingface.co/datasets/allenai/winogrande)

### Preparing Target Data

First, place the `train_dataset.npy` under directory `datasets`. Then run `data_prep.py`. This will complete the converting and the KNN sampling process.

The data used in 5 runs in our paper is under directory `datasets/exp/exp{0/1/2/3/4}` respectively.

## Comments

Our codebase is based on the following repo. Thanks for open-sourcing!

[Knowledge Unlearning](https://github.com/joeljang/knowledge-unlearning)

[llm_unlearn](https://github.com/kevinyaobytedance/llm_unlearn)
