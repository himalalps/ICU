python ./eval.py --exp "exp0" \
--model_name "EleutherAI/gpt-neo-125m" \
--tokenizer_name "EleutherAI/gpt-neo-125m" \
--gpt2_name "openai-community/gpt2" \
--bert_name "google-bert/bert-base-uncased" \
--prefix_length 200 --suffix_length 200 --target_length 200 \
--device "cuda:3" --batch_size 8 --num_workers 8 \
--dir "./result/test"