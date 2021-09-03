# BERT

There is a lot of BERT implementations open sourced:

- The [BERT repository by Google Research](https://github.com/google-research/bert) provide a comprehensive step-by-step guide in order to train and finetune BERT. It, however, don't provide multi-gpu/multinode support. 
- [Nvidia's DeepLearningExamples repository](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT) provide a impresive scalable implementation of BERT, but it focus on reproducing BERT's training, and, in order to train with your own texts, you have to dive deep into the source code. 
- [Hugginface's transformers examples](https://github.com/huggingface/transformers/tree/master/examples/pytorch/language-modeling) provides a masked language modelling training script that works out-of-the-box and scales to multi-gpu/multi-node. It, however, don't train with the Next Sentence Prediction objective necessary to train BERT.

We provide here an implementation that works out-of-the-box and is also scalable by adapting scripts from Google Research and Hugginface. As a bonus, our use of huggiface's transformers allows to use different pretrained BERT models as start point for the training. 

## Step-by-step guide to train your LM

### 1. Save BERT's vocab to file

create_pretraining_data.py will need the vocab on a file

```bash
python create_vocab.py --model_name_or_path bert-base-uncased --vocab_file data/vocab.txt
```

### 2. Create pretraining data
```bash
python src/pretraining/create_pretraining_data.py \
  --input_file=data/train.txt \
  --output_file=data/train_128.json \
  --vocab_file=data/vocab.txt \
  --do_lower_case=True \
  --do_whole_word_mask=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

The input for the create pretraining data script must be a text file where each line is a sentence and documents are separated by a blank line. For example:
```txt
This sentence belongs to document 1.
This sentence also belongs to document 1.

This sentence belongs to document 2.
This sentence also belongs to document 2.
```

Note that `max_predictions_per_seq` should be close to `masked_lm_prob * max_seq_length`.

For extra details, please consult: https://github.com/google-research/bert#pre-training-with-bert . We only made small changes to their create_pretraining_data.py script in order to change the output format.

### 3. Train BERT
As recomended by the the autors, training BERT is best done by first training with small sequences (128 tokens) and then with larger sequences (512 tokens). Since attention is quadratic with sentence length, this allows for more efficient learning. The autors recommend training 90% os the steps with 128-token sentences and the 10% remaining with 512-token sentences. 

To train with 128 tokens, on 4 GPUs, use:
```
python -m torch.distributed.run --nproc_per_node=4 run_bert.py \
	--model_name_or_path=bert-base-uncased \
	--train_file data/train_128.json \
	--validation_file=data/dev_128.json \
  --datasets_cache_dir=data/cache \
	--do_train=True \
	--do_eval=True \
	--evaluation_strategy=steps \
	--eval_steps=5000 \
	--per_device_train_batch_size=96 \
	--per_device_eval_batch_size=192 \
	--gradient_accumulation_steps=2 \
	--eval_accumulation_steps=24 \
	--learning_rate=5e-5 \
	--weight_decay=0.01 \
	--max_steps=90000 \
	--lr_scheduler_type=linear \
	--warmup_steps=900 \
	--logging_first_step=True \
	--save_strategy=steps \
	--save_steps=5000 \
	--dataloader_drop_last=True \
	--remove_unused_columns=False \
	--label_names labels next_sentence_label \
	--logging_dir=runs \
	--dataloader_num_workers=2 \
	--output_dir=data/models/checkpoints_128
```

Train also for 512-token sequences.

### 4. Evaluate

Evaluate for MLM objective only to obtain perplexity metrics:
```bash
python -m torch.distributed.run --nproc_per_node=4 run_bert.py \
	--model_name_or_path=data/models/checkpoints_512/checkpoint-10000 \
	--validation_file=data/test_512.json \
  --datasets_cache_dir=data/cache \
	--do_train=False \
	--do_eval=True \
	--per_device_eval_batch_size=48 \
	--eval_accumulation_steps=24 \
	--dataloader_drop_last=False \
	--remove_unused_columns=False \
	--label_names labels \
	--logging_dir=runs_eval \
	--dataloader_num_workers=2 \
	--eval_only_mlm=True \
	--overwrite_cache=True \
	--overwrite_output_dir=True \
	--output_dir=temp
```

or to obtain NSP + MLM loss metrics:

```bash
python -m torch.distributed.run --nproc_per_node=4 run_bert.py \
  --model_name_or_path=data/models/checkpoints_512/checkpoint-10000 \
  --validation_file=data/test_512.json \
  --datasets_cache_dir=data/cache \
	--do_train=False \
	--do_eval=True \
	--per_device_eval_batch_size=48 \
	--eval_accumulation_steps=24 \
	--dataloader_drop_last=False \
	--remove_unused_columns=False \
	--label_names labels next_sentence_label \
	--logging_dir=runs_eval \
	--dataloader_num_workers=2 \
	--eval_only_mlm=False \
	--overwrite_cache=True \
	--overwrite_output_dir=True \
	--output_dir=temp
```