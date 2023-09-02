# DEMO_NAME="newton_einstein"
# DEMO_NAME_GENDERS=(
#     [0]="role1 girl"
#     [1]="role2 girl"
#     [2]="role3 girl"
#     [3]="role4 girl"
#     [4]="role5 girl"
#     [5]="role6 boy"
#     [6]="role7 boy"
#     [7]="role8 boy"
#     [8]="role9 boy"
#     [9]="role10 boy"
# )
DEMO_NAME_GENDERS=(
    [0]="Nahida girl"
    [1]="6002903 girl"
    [2]="Elsa girl"
)
# DEMO_NAME_GENDERS=(
#     [0]="Nahida_Elsa girl"
# )
# DEMO_NAME_GENDERS=(
#     [0]="role1 girl"
# )


CKPTS=(
    [0]="115000"
)
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
#     [0]="5000"
#     [1]="10000"
#     [2]="15000"
#     [3]="20000"
#     [4]="25000"
#     [5]="30000"
#     [6]="35000"
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

# CAPTIONS=(
#     [0]="a GENDER <|image|>, solo, sitting in the classroom"
#     [1]="a GENDER <|image|>, solo, standing in the library"
#     [2]="a GENDER <|image|>, solo, lying on the bed"
#     [3]="a GENDER <|image|>, solo, running in the gym"
#     [4]="a GENDER <|image|>, solo, holding flower in a forest with trees"
#     [5]="a GENDER <|image|>, solo, victory pose, on the stage"
# )
CAPTIONS=(
    [0]="a GENDER <|image|> is reading book"
)
# CAPTIONS=(
#     [0]="a GENDER <|image|> and a GENDER <|image|> are reading book together"
# )
# CAPTIONS=(
#     [0]="a GENDER <|image|>, solo, on the playground"
# )


# POSE_FOLDER='pose_512_288'
POSE_FOLDER='pose_512_768'
# POSE_FOLDER='pose_1024_576'
# POSES=(
#     [0]="001_stand/000000006_pose.jpg"
#     [1]="002_hold/000010006_pose.jpg"
#     [2]="003_sit/000020732_pose.jpg"
#     [3]="005_walk/000030054_pose.jpg"
#     [4]="007_play/000040005_pose.jpg"
#     [5]="008_look/000050028_pose.jpg"
#     [6]="010_ride/000070151_pose.jpg"
#     [7]="014_lay/000100001_pose.jpg"
#     [8]="016_do/000110006_pose.jpg"
#     [9]="020_run/000140007_pose.jpg"
# )
POSES=(
    [0]="001_xxx/pose.png"
)


demo_name_gender_start_id=0
ckpt_start_id=0
caption_start_id=0
pose_start_id=0

for (( i="$demo_name_gender_start_id"; i<${#DEMO_NAME_GENDERS[@]}; i++ )); do
    IFS=' ' read -ra row <<< "${DEMO_NAME_GENDERS[i]}"
    DEMO_NAME="${row[0]}"
    GENDER="${row[1]}"
    for (( j="$ckpt_start_id"; j<${#CKPTS[@]}; j++ )); do
        CKPT="${CKPTS[j]}"
        for (( k="$caption_start_id"; k<${#CAPTIONS[@]}; k++ )); do
            CAPTION="${CAPTIONS[k]}"
            NEW_CAPTION="$(echo "$CAPTION" | sed "s/GENDER/$GENDER/g")"
            for (( x="$pose_start_id"; x<${#POSES[@]}; x++ )); do
                POSE="${POSES[x]}"
                SAVE_POSE=$(echo "$POSE" | sed 's/\// /g')
                echo "DEMO_NAME:${DEMO_NAME} GENDER:${GENDER} CKPT:${CKPT} NEW_CAPTION:${NEW_CAPTION} POSE_FOLDER:${POSE_FOLDER} POSE:${POSE}"

                CUDA_VISIBLE_DEVICES=0 accelerate launch \
                    --config_file /nas40/chenyu.liu/cache/huggingface/accelerate/second_config.yaml \
                    --mixed_precision=fp16 \
                    fastcomposer/inference_controlnet.py \
                    --pretrained_model_name_or_path /nas40/chenyu.liu/Models/anything-v3.0 \
                    --finetuned_model_path /nas40/chenyu.liu/fastcomposer_release_danbooru/fastcomposer-main/models_blip2_captions/anything-v3.0/danbooru/postfuse-localize-danbooru-1_5-1e-5/checkpoint-${CKPT} \
                    --test_reference_folder data/${DEMO_NAME} \
                    --test_caption "${NEW_CAPTION}" \
                    --pose_image_path /nas40/chenyu.liu/Datasets/pose/"${POSE_FOLDER}"/"${POSE}" \
                    --output_dir outputs/${DEMO_NAME}/${CKPT}/"${NEW_CAPTION}"/"${SAVE_POSE}" \
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
            done
        done
    done
done