


    EXP_NAME=outputs/qwen25_math7b_grpo_baseline \
    WANDB_RUN_ID=y52m6blw \
    WANDB_RESUME=allow \
    bash sh/baseline/grpo/grpo.sh \
    trainer.total_epochs=10 \
    trainer.resume_mode=auto

