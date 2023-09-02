# DEMO_NAME="newton_einstein"
DEMO_NAME_GENDERS=(
    [0]="role1 girl"
    [1]="role2 girl"
    [2]="role3 girl"
    [3]="role4 girl"
    [4]="role5 girl"
    [5]="role6 boy"
    [6]="role7 boy"
    [7]="role8 boy"
    [8]="role9 boy"
    [9]="role10 boy"
)
# DEMO_NAME_GENDERS=(
#     [0]="Nahida girl"
#     [1]="6002903 girl"
#     [2]="Elsa girl"
# )
# DEMO_NAME_GENDERS=(
#     [0]="Nahida_Elsa girl"
# )
# DEMO_NAME_GENDERS=(
#     [0]="daji girl"
# )


CKPTS=(
    [0]="220000"
)
# CKPTS=(
#     [0]="105000"
#     [1]="110000"
#     [2]="115000"
# )
# CKPTS=(
#     [0]="90000"
#     [1]="100000"
#     [2]="110000"
#     [3]="120000"
#     [4]="130000"
#     [5]="140000"
#     [6]="150000"
# )
# CKPTS=(
#     [0]="120000"
#     [1]="130000"
#     [2]="140000"
#     [3]="150000"
#     [4]="160000"
#     [5]="170000"
#     [6]="180000"
#     [7]="190000"
#     [8]="200000"
#     [9]="210000"
# )
# CKPTS=(
#     [0]="65000"
#     [1]="70000"
#     [2]="75000"
#     [3]="80000"
#     [4]="85000"
#     [5]="90000"
#     [6]="95000"
#     [7]="100000"
# )

CAPTIONS=(
    [0]="a GENDER <|image|>, solo, sitting in the classroom"
    [1]="a GENDER <|image|>, solo, standing in the library"
    [2]="a GENDER <|image|>, solo, lying on the bed"
    [3]="a GENDER <|image|>, solo, running in the gym"
    [4]="a GENDER <|image|>, solo, holding flower in a forest with trees"
    [5]="a GENDER <|image|>, solo, victory pose, on the stage"
)
# CAPTIONS=(
#     [0]="a GENDER <|image|> is reading book"
# )
# CAPTIONS=(
#     [0]="a GENDER <|image|> and a GENDER <|image|> are reading book together"
# )


demo_name_gender_start_id=0
ckpt_start_id=0
caption_start_id=0

for (( i="$demo_name_gender_start_id"; i<${#DEMO_NAME_GENDERS[@]}; i++ )); do
    IFS=' ' read -ra row <<< "${DEMO_NAME_GENDERS[i]}"
    DEMO_NAME="${row[0]}"
    GENDER="${row[1]}"
    for (( j="$ckpt_start_id"; j<${#CKPTS[@]}; j++ )); do
        CKPT="${CKPTS[j]}"
        for (( k="$caption_start_id"; k<${#CAPTIONS[@]}; k++ )); do
            CAPTION="${CAPTIONS[k]}"
            NEW_CAPTION="$(echo "$CAPTION" | sed "s/GENDER/$GENDER/g")"
            
            echo "DEMO_NAME:${DEMO_NAME} GENDER:${GENDER} CKPT:${CKPT} NEW_CAPTION:${NEW_CAPTION}"

            CUDA_VISIBLE_DEVICES=0 accelerate launch \
                --config_file /oss/comicai/chenyu.liu/cache/huggingface/accelerate/second_config.yaml \
                --mixed_precision=fp16 \
                fastcomposer/inference.py \
                --pretrained_model_name_or_path /oss/comicai/chenyu.liu/Models/anything-v3.0 \
                --finetuned_model_path /oss/comicai/chenyu.liu/fastcomposer_release_danbooru/fastcomposer-main/models_blip2_captions/anything-v3.0/danbooru/postfuse-localize-danbooru-1_5-1e-5/checkpoint-${CKPT} \
                --test_reference_folder /oss/comicai/chenyu.liu/Datasets/person/${DEMO_NAME} \
                --test_caption "${NEW_CAPTION}" \
                --output_dir outputs/${DEMO_NAME}/${CKPT}/"${NEW_CAPTION}" \
                --mixed_precision fp16 \
                --image_encoder_type clip \
                --image_encoder_name_or_path openai/clip-vit-large-patch14 \
                --num_image_tokens 1 \
                --max_num_objects 2 \
                --object_resolution 224 \
                --generate_height 288 \
                --generate_width 512 \
                --num_images_per_prompt 4 \
                --num_rows 1 \
                --seed 42 \
                --guidance_scale 5 \
                --inference_steps 50 \
                --start_merge_step 0 \
                --no_object_augmentation
        done
    done
done