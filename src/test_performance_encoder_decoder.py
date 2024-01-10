import os
import sys
import time
import json
import logging
import argparse
from tqdm import tqdm

from datasets import load_dataset

from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import GenerationConfig

import torch
from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="google/flan-t5-base")

    parser.add_argument("--dataset", type=str, default="cnn_dailymail")
    parser.add_argument("--num_samples", type=int, default=100000)
    
    parser.add_argument("--loading_mode", type=str, default="fp16")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--repetition_penalty", type=float, default=1.)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--context_aware_decoding_alpha", type=float, default=0.)

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--save_output", action="store_true")
    parser.add_argument("--output_fname", type=str, default="")

    parser.add_argument("--logging", type=str, default="./default_encoder_decoder.log")

    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        filename=args.logging, 
        filemode='a',
        )
    logger = logging.getLogger(__name__)
    for k, v in vars(args).items():
        logger.info(f"{k} -> {v}")
    logger.info(f"\n")

    train_set, validation_set, test_set = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token, tokenizer.pad_token_id = tokenizer.eos_token, tokenizer.eos_token_id

    test_set = pretokenize(test_set[:args.num_samples], tokenizer, args.max_input_length)
    test_set = [[template_input_encoder_decoder(row, args.dataset), row[-1]] for row in test_set]
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
        )
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    generation_config = GenerationConfig(
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens, 
        early_stopping=False,  # early stopping is only effective in beam search
        do_sample=args.do_sample,
        num_beams=args.num_beams,
        top_k=args.top_k,
        top_p=args.top_p,  # my understanding is that top-k and top-p can be combined, source from https://huggingface.co/blog/how-to-generate
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        context_aware_decoding_alpha=args.context_aware_decoding_alpha,
        bos_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    dataset = test_set
    num_batch = len(dataset)//args.batch_size+1 if len(dataset)%args.batch_size!=0 else len(dataset)//args.batch_size

    # if args.context_aware_decoding_alpha > 0.:
    #     null_input = get_null_input_encoder_decoder(args.dataset)

    logger.info("start decoding!")
    predictions, references, documents = [], [], []
    start_time = time.time()
    for batch_idx, _ in enumerate(range(num_batch)):
        batch = dataset[args.batch_size*batch_idx: args.batch_size*(batch_idx+1)]
        batch_input, batch_reference = [row[0] for row in batch], [row[1] for row in batch]
        tokenized_input = tokenizer(batch_input, return_tensors="pt", max_length=512, padding=True, truncation=True)
        if args.context_aware_decoding_alpha > 0.:
            batch_null_input = [get_null_input_decoder(row, args.dataset) for row in batch]
            tokenized_null_input = tokenizer(batch_null_input, return_tensors="pt", padding=True)
            with torch.no_grad():
                output = model.generate(
                    input_ids=tokenized_input.input_ids.to(DEVICE),
                    null_inputs=tokenized_null_input.input_ids.to(DEVICE),
                    attention_mask=tokenized_input.attention_mask.to(DEVICE),
                    generation_config=generation_config,
                )
        else:
            with torch.no_grad():
                output = model.generate(
                    input_ids=tokenized_input.input_ids.to(DEVICE),
                    attention_mask=tokenized_input.attention_mask.to(DEVICE),
                    generation_config=generation_config
                )
        predictions.extend(tokenizer.batch_decode(output, skip_special_tokens=True, reduce_tokenization_space=True))
        references.extend(batch_reference)
        documents.extend(batch_input)
    
    print(f"total decoding takes {time.time()-start_time:.1f} seconds!")
    del model  # offload model to avoid memory spike

    assert len(predictions)==len(references), "mismatched shape, force exit!"
    evaluator = Evaluator()

    result_dict = evaluator.evaluate(predictions, references, documents)
    for k, v in result_dict.items():
        logger.info(f"{k} -> {v*100:.1f}")
    logger.info("\n")

    if args.save_output:
        if not args.output_fname:
            model_name = config._name_or_path.split("/")[-1]
            args.output_fname = f"{model_name}_max_new_tokens_{args.max_new_tokens}_topk_{args.top_k}_topp_{args.top_p}_alpha_{args.context_aware_decoding_alpha}.jsonl"
            args.output_fname = os.path.join("./generation", args.output_fname)

        with open(args.output_fname, "w") as fout:
            for i in range(len(predictions)):
                json_line = {"prediction": predictions[i], "reference:": references[i]}
                json.dump(json_line, fout)
                fout.write("\n")
        fout.close()
        logger.info(f"generation file -> {args.output_fname}\n")