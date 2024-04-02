A repo for cool image, text retrieval using OpenAI's [ClIP](https://arxiv.org/abs/2103.00020) model. For information about how CLIP works see my
article on [Medium](https://medium.com/@paluchasz/understanding-openais-clip-model-6b52bade3fa3).

## Data
Using the following Flickr 8K dataset from [Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k). It contains a variety of images each paired
with 5 different captions.

## Running Locally
The first step is to pre-compute the image embeddings with the `image_text_retrieval/scripts/pre_compute_embeddings.py` script.
