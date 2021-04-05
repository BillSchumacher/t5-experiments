# t5-experiments

## Pre-training
### T5-small baseline
```bash
export CUDA_VISIBLE_DEVICES=4,5; horovodrun --gloo -np 2 python run_t5_pretraining.py --batch_size 32 --gradient_accumulation_steps 2 --save_interval 100000 --log_interval 500 --iters 1100000 --data_path ~/data/ThePile/Wikipedia/preprocessed_shards --model_path ./runs/small_wiki_bs_128 --input_seq_len 512 --target_seq_len 192 --lr 5e-05 --model_cfg ./t5configs/t5-small.json
```

### T5-base with custom layers:
and continue interrupted training
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3; horovodrun --gloo -np 4 python run_t5_pretraining.py --batch_size 8 --gradient_accumulation_steps 4 --save_interval 75000 --log_interval 500 --iters 1000000 --data_path ~/data/ThePile/Wikipedia/preprocessed_shards --model_path ./runs/base_wiki_enc_only_cdq_fixed_pos_wo_tanh --input_seq_len 512 --target_seq_len 192 --lr 5e-05 --model_cls modeling_t5:T5ForConditionalGeneration --model_cfg t5configs/t5-base-only-cdQ.json --init_checkpoint ./runs/base_wiki_enc_only_cdq_fixed_pos_wo_tanh/model_150000.pth
```

## Fine-tuning with DeepPavlov
`python -m deeppavlov train config_name`

Gradient accumulation for `dp:T5Text2TextModel`, e.g.:
- `batch_size`: 32
- `sub_batch_size`: 16

means that full batch of size `batch_size` will be splited on two sub-batches of size `sub_batch_size` to accumulate their gradients.

### Fine-tuning on GLUE
Base configuration files are at `./dp_configs/glue`

**TBD**: prepare script for fine-tuning on all GLUE tasks for pre-trained model

#### GLUE mixture from T5
config: `./dp_configs/glue/glue_mixture.json`

Use `save_every_n_batches` parameter to save the model, set `metrics: []` and `evaluation_targets: []`.

Evaluation for all checkpoints in `mixture_model` folder, saves best checkpoints and evaluation results:
```bash
export CUDA_VISIBLE_DEVICES=0; python evaluate_mixture_model.py \
        --mixture_model ./runs/small_wiki_bs_128/glue/mixture/bs_128 \
        --pretrained_checkpoint ./runs/small_wiki_bs_128/model_1100000.pth \
        --task_configs_path ./dp_configs/glue \
        --save_best
```

### Prepare submission for GLUE Leaderboard:
**TBD**

#### QQP
QQP is currently not available via tfds: https://github.com/tensorflow/datasets/pull/3031

to hot-fix this go to the source code of installed tfds `tensorflow_datasets/text/glue.py:215` and replace QQP data url with https://dl.fbaipublicfiles.com/glue/data/QQP.zip
