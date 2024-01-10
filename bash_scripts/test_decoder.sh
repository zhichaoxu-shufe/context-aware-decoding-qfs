DEVICE=1

CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
--model_name_or_path ../models/mistral-7b \
--num_samples 1000 \
--batch_size 4 \
--context_aware_decoding_alpha 0.15 \
--max_input_length 512 \
--min_new_tokens 30 \
--max_new_tokens 70 \
--save_output \
--dataset cnn_dailymail \
--do_sample 

CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
--model_name_or_path ../models/mistral-7b-instruct \
--num_samples 1000 \
--batch_size 4 \
--context_aware_decoding_alpha 0.15 \
--max_input_length 512 \
--min_new_tokens 50 \
--max_new_tokens 70 \
--save_output \
--dataset cnn_dailymail \
--do_sample 

CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
--model_name_or_path ../models/mpt-7b-instruct \
--num_samples 2000 \
--batch_size 4 \
--context_aware_decoding_alpha 0. \
--max_input_length 512 \
--min_new_tokens 30 \
--max_new_tokens 70 \
--save_output \
--dataset cnn_dailymail \
--do_sample 

CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
--model_name_or_path ../models/mpt-7b-instruct \
--num_samples 2000 \
--batch_size 4 \
--context_aware_decoding_alpha 0.15 \
--max_input_length 512 \
--min_new_tokens 30 \
--max_new_tokens 70 \
--save_output \
--dataset cnn_dailymail \
--do_sample 

CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
--model_name_or_path ../models/mpt-7b-instruct \
--num_samples 2000 \
--batch_size 4 \
--context_aware_decoding_alpha 0.3 \
--max_input_length 512 \
--min_new_tokens 30 \
--max_new_tokens 70 \
--save_output \
--dataset cnn_dailymail \
--do_sample 

CUDA_VISIBLE_DEVICES=$DEVICE python src/test_performance_decoder.py \
--model_name_or_path ../models/mpt-7b-instruct \
--num_samples 2000 \
--batch_size 4 \
--context_aware_decoding_alpha 0.5 \
--max_input_length 512 \
--min_new_tokens 30 \
--max_new_tokens 70 \
--save_output \
--dataset cnn_dailymail \
--do_sample 