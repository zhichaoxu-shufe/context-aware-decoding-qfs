DEVICE=1
MODEL=google/flan-t5-xxl
DATASET=xsum
LOGGING=flan-t5-xsum.log

CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_encoder_decoder.py \
--model_name_or_path $MODEL \
--num_samples 2000 \
--batch_size 8 \
--context_aware_decoding_alpha 0.0 \
--max_input_length 512 \
--min_new_tokens 20 \
--max_new_tokens 50 \
--save_output \
--dataset $DATASET \
--do_sample \
--logging $LOGGING

CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_encoder_decoder.py \
--model_name_or_path $MODEL \
--num_samples 2000 \
--batch_size 8 \
--context_aware_decoding_alpha 0.15 \
--max_input_length 512 \
--min_new_tokens 20 \
--max_new_tokens 50 \
--save_output \
--dataset $DATASET \
--do_sample \
--logging $LOGGING

CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_encoder_decoder.py \
--model_name_or_path $MODEL \
--num_samples 2000 \
--batch_size 8 \
--context_aware_decoding_alpha 0.3 \
--max_input_length 512 \
--min_new_tokens 20 \
--max_new_tokens 50 \
--save_output \
--dataset $DATASET \
--do_sample \
--logging $LOGGING

CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_encoder_decoder.py \
--model_name_or_path $MODEL \
--num_samples 2000 \
--batch_size 8 \
--context_aware_decoding_alpha 0.5 \
--max_input_length 512 \
--min_new_tokens 20 \
--max_new_tokens 50 \
--save_output \
--dataset $DATASET \
--do_sample \
--logging $LOGGING