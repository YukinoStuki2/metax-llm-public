MODEL_DIR=${MODEL_DIR:-./model/merged}
python eval_local.py --which basic --strip_q_suffix --model_dir_for_tokenizer "$MODEL_DIR"
