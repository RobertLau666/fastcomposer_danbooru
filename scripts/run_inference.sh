CAPTION="a girl <|image|> is reading book"
# DEMO_NAME="newton_einstein"
# DEMO_NAME="role_data_336k_pre" # 6002903
# DEMO_NAME="role_genshin" # Nahida
DEMO_NAME="other" # Elsa

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file /nas40/chenyu.liu/cache/huggingface/accelerate/second_config.yaml \
    --mixed_precision=fp16 \
    fastcomposer/inference.py \
    --pretrained_model_name_or_path /nas40/chenyu.liu/Models/anything-v3.0 \
    --finetuned_model_path /nas40/chenyu.liu/fastcomposer_release_danbooru/fastcomposer-main/models/anything-v3.0/danbooru/postfuse-localize-danbooru-1_5-1e-5/checkpoint-90000 \
    --test_reference_folder data/${DEMO_NAME} \
    --test_caption "${CAPTION}" \
    --output_dir outputs/${DEMO_NAME}/Elsa \
    --mixed_precision fp16 \
    --image_encoder_type clip \
    --image_encoder_name_or_path openai/clip-vit-large-patch14 \
    --num_image_tokens 1 \
    --max_num_objects 2 \
    --object_resolution 224 \
    --generate_height 512 \
    --generate_width 512 \
    --num_images_per_prompt 10 \
    --num_rows 1 \
    --seed 42 \
    --guidance_scale 5 \
    --inference_steps 50 \
    --start_merge_step 10 \
    --no_object_augmentation
