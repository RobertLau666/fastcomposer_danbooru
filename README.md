该项目基于[FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention](https://github.com/mit-han-lab/fastcomposer)，训练采用danbooru数据，经过blip2提取图片的prompt

## 准备数据
1. 先运行generate_blip2_captions_danbooru_.py，生成captions（用55 node2 tmux4）
2. 再运行file_process_.ipynb的“instantbooth/data_336k_pre数据处理（for release）”的“生成xxxxxxx_blip2_captions.json文件”

## 修改
修改run_training.sh（相比于第一次训练的run_training.sh）
变慢后，若中断，只需按一次Ctrl+C，静等停止，上拉框'resource_tracker: There appear to be %d '消失，wandb的绿点消失，等2h信号量释放？再重训就好（跟调整bsz、main_process_port大小没关系），一旦正常速度训练时，除了关闭两个终端，不要改变default_config.yaml（其他测试额外指定一个second_config.yaml文件就行了）和项目代码，包括不用删除wandb无关进程，否则会变卡住
其实也许没那么多道道，就是需要冷却？多试几次就好了
去掉该参数--main_process_port 11185 \ 就好了（或与参数无关）
不同训练版本要更换此参数值，大一些（如11195），否则主进程会在不同卡之间切换，耗时，会在每隔20-40steps变慢；
网上说调小bsz，尝试10->5，然后静等走step（尽管很慢），然后Ctrl+C，再调成10，就好了https://stackoverflow.com/questions/64515797/there-appear-to-be-6-leaked-semaphore-objects-to-clean-up-at-shutdown-warnings-w
--logging_dir logs_blip2_captions/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
--output_dir models_blip2_captions/${MODEL}/${DATASET_NAME}/${WANDB_NAME} \
--train_batch_size 12 \
--keep_interval 5000 \ #10000 每隔5k step保存一次
修改data.py
两处##".json"改为"_blip2_captions.json"

## 训练
bash scripts/run_training.sh

## 测试（用55 node2 tmux3，测试完显示完）
修改scripts/run_inference_batch.sh（相比于第一次训练的run_inference_batch.sh）
三个变量
97下，移动ckpt到oss下：
cp -r /ckpt_saved/models_blip2_captions/anything-v3.0/danbooru/postfuse-localize-danbooru-1_5-1e-5/checkpoint-220000 /oss/comicai/chenyu.liu/fastcomposer_release_danbooru/fastcomposer-main/models_blip2_captions/anything-v3.0/danbooru/postfuse-localize-danbooru-1_5-1e-5
--finetuned_model_path /nas40/chenyu.liu/fastcomposer_release_danbooru/fastcomposer-main/models_blip2_captions/anything-v3.0/danbooru/postfuse-localize-danbooru-1_5-1e-5/checkpoint-${CKPT} \
图像参数
bash scripts/run_inference_batch.sh

## 显示（用84）：
fastcomposer/show_ckpt_img_.ipynb

## 备份
97的/ckpt_saved文件夹下备份了115000
nas的fastcomposer_release_danbooru/fastcomposer-main项目下备份了5000-115000
oss的fastcomposer_release_danbooru/fastcomposer-main项目下备份了5000-115000

## +ControlNet
修改utils.py:
##添加参数--pose_image_path
inference_controlnet.py:
inference.py复制为inference_controlnet.py
    添加image = load_image(
        # '/nas40/chenyu.liu/Tests_/pose.png'
        args.pose_image_path
    )
run_inference_batch_controlnet.sh:
run_inference_batch.sh复制为run_inference_batch_controlnet.sh
添加POSES相关
改为fastcomposer/inference_controlnet.py \
bash run_inference_batch_controlnet.sh

## 显示（用84）：
fastcomposer/show_ckpt_img_.ipynb


# FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention [[website](https://fastcomposer.mit.edu/)] [[demo](https://fastcomposer.hanlab.ai)][[replicate api](https://replicate.com/cjwbw/fastcomposer)]

![multi-subject](figures/multi-subject.png)

## Abstract

Diffusion models excel at text-to-image generation, especially in subject-driven generation for personalized images. However, existing methods are inefficient due to the subject-specific fine-tuning, which is computationally intensive and hampers efficient deployment. Moreover, existing methods struggle with multi-subject generation as they often blend features among subjects. We present FastComposer which enables efficient, personalized, multi-subject text-to-image generation without fine-tuning. FastComposer uses subject embeddings extracted by an image encoder to augment the generic text conditioning in diffusion models, enabling personalized image generation based on subject images and textual instructions with only forward passes. To address the identity blending problem in the multi-subject generation, FastComposer proposes cross-attention localization supervision during training, enforcing the attention of reference subjects localized to the correct regions in the target images. Naively conditioning on subject embeddings results in subject overfitting. FastComposer proposes delayed subject conditioning in the denoising step to maintain both identity and editability in subject-driven image generation. FastComposer generates images of multiple unseen individuals with different styles, actions, and contexts. It achieves 300x-2500x speedup compared to fine-tuning-based methods and requires zero extra storage for new subjects. FastComposer paves the way for efficient, personalized, and high-quality multi-subject image creation.


## Usage

### Environment Setup

```bash
conda create -n fastcomposer python
conda activate fastcomposer
pip install torch torchvision torchaudio
pip install transformers accelerate datasets evaluate diffusers==0.16.1 xformers triton scipy clip gradio

python setup.py install
```

### Download the Pre-trained Models

```bash
mkdir -p model/fastcomposer ; cd model/fastcomposer
wget https://huggingface.co/mit-han-lab/fastcomposer/resolve/main/pytorch_model.bin
```

### Gradio Demo

We host a demo [here](https://fastcomposer.hanlab.ai/). You can also run the demo locally by 

```bash   
python demo/run_gradio.py --finetuned_model_path model/fastcomposer/pytorch_model.bin  --mixed_precision "fp16"
```

### Inference

```bash
bash scripts/run_inference.sh
```

### Training

Prepare the FFHQ training data:
  
```bash 
cd data
wget https://huggingface.co/datasets/mit-han-lab/ffhq-fastcomposer/resolve/main/ffhq_fastcomposer.tgz
tar -xvzf ffhq_fastcomposer.tgz
```

Run training:

```bash
bash scripts/run_training.sh
```

## TODOs

- [x] Release inference code
- [x] Release pre-trained models
- [x] Release demo
- [x] Release training code and data
- [ ] Release evaluation code and data

## Citation

If you find FastComposer useful or relevant to your research, please kindly cite our paper:

```bibtex
@article{xiao2023fastcomposer,
            title={FastComposer: Tuning-Free Multi-Subject Image Generation with Localized Attention},
            author={Xiao, Guangxuan and Yin, Tianwei and Freeman, William T. and Durand, Frédo and Han, Song},
            journal={arXiv},
            year={2023}
          }
```


