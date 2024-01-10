This is the code base for our technical report <b>Context-aware Decoding Reduces Hallucination in Query-focused Summarization</b>, https://arxiv.org/pdf/2312.14335.pdf

#### Required packages
```
sklearn
torchmetrics
transformers
datasets
evaluate
```

#### Install huggingface library
follow instructions in https://huggingface.co/docs/transformers/installation and do <b>Editable Install</b>

#### Add context-aware decoding
replace `transformers/src/transformers/generation/utils.py` with `generation/utils.py`

#### Reproduce the experimental results
- Download the required datasets from [this google drive link](https://drive.google.com/drive/folders/1pFCmEBX8cUM3OsG-qDb6H1O8Uy3Dahr_?usp=sharing)

- Change the path in `./src/utils.py` accordingly

- Run sample bash scripts in `./src/bash_scripts`

- A detailed list of arguments can be found at `src/test_performance_decoder.py` and `src/test_performance_encoder_decoder.py`


#### Correspondence
Contact zhichao.xu@utah.edu if you have trouble running the code

##### Citations
```
@article{shi2023trusting,
  title={Trusting Your Evidence: Hallucinate Less with Context-aware Decoding},
  author={Shi, Weijia and Han, Xiaochuang and Lewis, Mike and Tsvetkov, Yulia and Zettlemoyer, Luke and Yih, Scott Wen-tau},
  journal={arXiv preprint arXiv:2305.14739},
  year={2023}
}
@article{xu2023context,
  title={Context-aware Decoding Reduces Hallucination in Query-focused Summarization},
  author={Xu, Zhichao},
  journal={arXiv preprint arXiv:2312.14335},
  year={2023}
}
```
