# Intro to DL Capstone Project: Artistic Style Transfer

# About
This is the repository of Group 25 for the capstone project of IT3320E - Intro to Deep Learning course. 

|       Name       | Student ID|
| ---------------- | --------- |
| Nguyen Duc An   | 20225432  |
| Mai Viet Bao  | 20225474  |
| Nguyen Nhu Giap | 20225441  |
| Pham Minh Hieu | 20220062  |
| Truong Tuan Vinh  | 20225464  |


# Usage

To use the code, first, clone this repository:

```.bash
git clone https://github.com/NguyenDucAnforwork/DL_project.git
```

Make sure to change your working directory to this repository.

```.bash
cd DL_project
```

Install all required packages by running this following command. You may want to create an additional virtual environment to avoid dependency conflicts.

```.bash
pip install -r requirements.txt
```

Next, download the checkpoints from this [link](https://husteduvn-my.sharepoint.com/:f:/g/personal/hieu_pm220062_sis_hust_edu_vn/Elcb7-0vRbFEt3BdGp_Pb0YBhxcdu7hoF3c556baxfBFgQ?e=IFe2uH). Create a new folder named `checkpoints` in the current working directory and **make sure to save all of the checkpoints there**. 

## Inference
### For the original Neural Style Transfer

Given a content image, a style image, one can infer using the following command:
```.bash
python3 transfer_original.py --content path_to_content_img --style path_to_style_img
```
For example, you can run this command for inference using content and style images provided in the `test_img` folder.

```.bash
python3 transfer_original.py --content test_img/content.jpg --style test_img/style.jpg
```
The output image is saved as `output_nst_original.png` in your current directory.

Content image             |  Style image          | Output image
:-------------------------:|:-------------------------:|:-----------------------:
![content](https://github.com/user-attachments/assets/087de1bc-2386-4e85-970b-176c6fbf6b37)|![style](https://github.com/user-attachments/assets/34a98877-13e2-469c-a16d-b9ebf8f79d61)|![output_nst_original](https://github.com/user-attachments/assets/b0c01666-39f9-4fb6-9169-d047743d8d5a)

### For AdaIn
Similar to the original Neural Style Transfer, make sure to prepare a content image and a style image, and run the following command:

```.bash
python3 AdaIn.py --content path_to_content_img --style path_to_style_img --model_path checkpoints/adain_model
```

One can run the following command for inference with our given examples.

```.bash
python3 AdaIn.py --content test_img/content.jpg --style test_img/style.jpg --model_path checkpoints/adain_model
```
The output image is saved as `output_adain.jpg` in your current directory.

### For CycleGAN
For this model, you only need to prepare an original photo for inference. Run the following command:
```.bash
python3 infer_cyclegan.py --model_path path_to_your_model --image_path path_to_your_img --device your_device
```
The supported devices are `cuda` and `cpu`. We recommend choosing `cuda` for faster inference; if `cuda` is not avalable in your working environment, consider `cpu` instead. For the model path, we provide five CycleGAN models, two of which correspond to the style of van Gogh and each of the remaining corresponds to Cézanne, Monet and Ukiyo-e, respectively:
- `checkpoints/cyclegan_vangogh_resnet_70_epochs.ckpt`
- `checkpoints/cyclegan_vangogh_unet_70_epochs.ckpt`
- `checkpoints/cyclegan_cezanne_unet_300_epochs.ckpt`
- `checkpoints/cyclegan_monet_unet_250_epochs.ckpt`
- `checkpoints/G_BA_20_epoch.pth` (which corresponds to the style of Ukiyo-e)

For example, you can transfer an example given in `test_img` folder to van Gogh's style as follows:
```
python3 infer_cyclegan.py --model_path checkpoints/cyclegan_vangogh_resnet_70_epochs.ckpt --image_path test_img/input_cyclegan.jpg 
```

Input image             | Output image
:-------------------------:|:-----------------------:
![input_cyclegan](https://github.com/user-attachments/assets/abcc2c8c-ce51-4a83-b482-b20db3317ce1)|![output_cyclegan](https://github.com/user-attachments/assets/f86576f8-473e-49e6-bde6-14b5e85e35d5)

The output image is saved as `output_cyclegan.png` in your current directory.

## App
Looking to explore image style transfer? Try our application by running this following command:
```
python3 app.py
```
After that, access the URL (printed in the terminal) and get started!

![image](https://github.com/user-attachments/assets/1ab08304-08ec-4c7f-9f1d-f95687e2d6be)







