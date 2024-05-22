python ./train_neo_meanwhile_update.py \
--exp "exp0" --model_name "EleutherAI/gpt-neo-125m" \
--tokenizer_name "EleutherAI/gpt-neo-125m" \
--gpt2_name "openai-community/gpt2" \
--bert_name "google-bert/bert-base-uncased" \
--prefix_length 200 --suffix_length 200 --target_length 200 \
--device "cuda:7" --batch_size 8 --num_workers 8 --lr 5e-6 \
--uw 1.0 --lw 0.5 --kl 1.0 --f1 0.3 --bleu 0.01 --acc 0.5994 \
--el 0.0499 --dir "result/test"