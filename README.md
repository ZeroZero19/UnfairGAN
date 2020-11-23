# UnfairGAN: An Enhanced Generative Adversarial Network for Raindrop Removal from A Single Image
## Requirements
- python       3.6
- pytorch      1.0
- CUDA         8.0 or 9.0
## Download
[Download all datasets and pretrains](https://drive.google.com/drive/u/2/folders/1-uIc10i4ktPpfk-6f0rnenWS-e24wHJK?fbclid=IwAR3T1XSaw3t7V5lVOs2ggnbGsKGKlCW427SpslnZ7SQ4X_JLmKBJ9jN31Is)

## Usage
### 1. Run Test
#### a. Test our data
python test_ours_data.py
#### b. Test real rain
python test_real_rain.py
#### c. Test real rain
python test_raindrop.py
#### Optional
- --testset: input folder (select testset in testsets folder)
- --model : weight (select derain's method in weight folder)
## Reference
[1] [Attentive Generative Adversarial Network for Raindrop Removal from A Single Image](https://github.com/rui1996/DeRaindrop)

[2] [I Can See Clearly Now : Image Restoration via De-Raining](https://github.com/meton-robean/ICanSeeClearyNow_unofficial)

[3] [Image-to-Image Translation Using Conditional Adversarial Networks](https://github.com/mrzhu-cool/pix2pix-pytorch)

[4] [Richer Convolutional Features for Edge Detection](https://github.com/meteorshowers/RCF-pytorch)