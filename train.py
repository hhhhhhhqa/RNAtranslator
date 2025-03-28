import torch
import json
import os

from transformers import Trainer, TrainingArguments, TrainerCallback , get_cosine_schedule_with_warmup
from tokenizers import Tokenizer
from transformers.trainer_callback import TrainerControl, TrainerState
from accelerate import Accelerator


def train(args:object, wandb, model, train_dataset, eval_dataset)->None:
    training_args = TrainingArguments(
        
        # report_to="wandb",
        output_dir=args.results_dir + "/checkpoints",
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        #resume_from_checkpoint="/data6/sobhan/rllm/results/train/t5/run3_20240822-152114/checkpoints/checkpoint-141200", # Resume Training
        ignore_data_skip = False, # Resume Training
        prediction_loss_only=args.prediction_loss_only,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        eval_delay=args.eval_delay,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        log_level=args.log_level,
        log_level_replica=args.log_level_replica,
        log_on_each_node=args.log_on_each_node,
        logging_dir=args.logging_dir,
        logging_strategy=args.logging_strategy,
        logging_first_step=args.logging_first_step,
        logging_steps=args.logging_steps,
        logging_nan_inf_filter=args.logging_nan_inf_filter,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_on_each_node=args.save_on_each_node,
        use_cpu=args.use_cpu,
        seed=args.seed,
        data_seed=args.data_seed,
        jit_mode_eval=args.jit_mode_eval,
        use_ipex=args.use_ipex,
        bf16=args.bf16,
        fsdp=args.fsdp,
        # fsdp_min_num_params=args.fsdp_min_num_params,
        # fsdp_config=args.fsdp_config,
        fp16=False,
        fp16_opt_level=args.fp16_opt_level,
        fp16_backend=args.fp16_backend,
        bf16_full_eval=args.bf16_full_eval,
        fp16_full_eval=args.fp16_full_eval,
        tf32=args.tf32,
        local_rank=args.local_rank,
        ddp_backend=args.ddp_backend,
        tpu_num_cores=args.tpu_num_cores,
        dataloader_drop_last=args.dataloader_drop_last,
        eval_steps=args.eval_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        past_index=args.past_index,
        run_name=args.run_name,
        disable_tqdm=args.disable_tqdm,
        remove_unused_columns=args.remove_unused_columns,
        label_names=args.label_names,
        load_best_model_at_end=args.load_best_model_at_end
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    # trainer.train(resume_from_checkpoint=True) # Resume Training
    return model

