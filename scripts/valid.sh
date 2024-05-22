python ./valid.py \
--model_name "./result/test/EleutherAI/gpt-neo-125m_exp0_lr5e-06_uw1.0_lw0.5_kl1.0_epoch19_updateboth" \
--tokenizer_name "EleutherAI/gpt-neo-125m" \
--prefix_length 512 --suffix_length 512 --device "cuda:3" \
--batch_size 32 --num_workers 48 \
--dir "./result/test" --cache "./.cache"