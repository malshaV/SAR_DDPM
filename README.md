# SAR-DDPM 

Code for the paper [SAR despeckling using a Denoising Diffusion Probabilistic Model](https://arxiv.org/pdf/2206.04514.pdf), acepted at IEEE Geoscience and Remote Sensing Letters


## To  train the SAR-DDPM model:

- Download the weights 64x64 -> 256x256 upsampler from [here](https://github.com/openai/guided-diffusion).

- Create a folder ./weights and place the dowloaded weights in the folder.

- Specify the paths to your training data and validation data in ./scripts/sarddpm_train.py (line 23 and line 25)

- Use the following command to run the code (change the GPU number according to GPU availability):

```bash
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --large_size 256  --small_size 64 --learn_sigma True --noise_schedule linear --num_channels 192 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True" 
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=0 python scripts/sarddpm_train.py $MODEL_FLAGS
```