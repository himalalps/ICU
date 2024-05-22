deepspeed --include localhost:0,1,2,3,4,5 \
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