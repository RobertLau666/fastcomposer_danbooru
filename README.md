The project is based on [FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention](https://github.com/mit-han-lab/fastcomposer), training dataset use danbooru dataset, prompt use the prompt generated from the danbooru dataset detected by BLIP2 model

## Preparing Database
1. Prepare a folder "train_fastcomposer_data_336k_pre_release_danbooru_" for storing data, the data storage format is as follows:
```
| train_fastcomposer_data_336k_pre_release_danbooru_/
|---- 00000/
|---- image_ids_train.txt
```
2. Run generate_blip2_captions_danbooru_.py, the generated file "blip2_captions_danbooru_336k_.json" is used in step 3.
3. Run file_process_.ipynb的"instantbooth/data_336k_pre数据处理（for release）"的"生成xxxxxxx_blip2_captions.json文件"

## Revise
1. Revise run_training.sh
```
--logging_dir logs_blip2_captions/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
--output_dir models_blip2_captions/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
--train_batch_size 12 \
--keep_interval 5000 \ #10000 每隔5k step保存一次
```
2. Revise data.py
Change the "##" in both places ".json" to "blip2 captions.json"

## Train
```
bash scripts/run_training.sh
```

## Test
Revise "scripts/run_inference_batch.sh"
```
bash scripts/run_inference_batch.sh
```

## Show
Run fastcomposer/show_ckpt_img_.ipynb

## + ControlNet
1. Revise utils.py: 

    a. Add parameter--pose_image_path in "##"

2. Revise inference_controlnet.py: 

    a. Copy inference.py to inference_controlnet.py

    b. Add as follows:
    ```
      image = load_image(
          # '/nas40/chenyu.liu/Tests_/pose.png'
          args.pose_image_path
      )
    ```
3. Revise "run_inference_batch_controlnet.sh":

    a. Copy "run_inference_batch.sh" to "run_inference_batch_controlnet.sh"

    b. Add the parameters associated with "POSES"

    c. Revise "fastcomposer/inference_controlnet.py \\"
```
bash run_inference_batch_controlnet.sh
```

## Show
Run fastcomposer/show_ckpt_img_.ipynb