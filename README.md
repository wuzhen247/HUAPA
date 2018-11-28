# HUAPA
HUAPA is the proposed model in 《[Improving Review Representations with User Attention and Product Attention for Sentiment Classification](https://arxiv.org/abs/1801.07861)》, which is accepted by AAAI'18.

# Data
The original datasets are released by the paper [Tang et al., 2015]. [[Download]](http://ir.hit.edu.cn/%7Edytang/paper/acl2015/dataset.7z)

The embedding file. [[Download]](https://drive.google.com/drive/folders/19r6Bj-s0intWCSIdd1W3dqOPueAc_K2N)

# Train
For example, you can use the folowing command to train HUAPA in the dataset imdb:
> python train.py --n_class 10 --dataset imdb

The best model will be saved in the folder "../checkpoints/imdb/timestamp". The timestamp is server time.

# Test
For example, you can use the folowing command to test HUAPA in the dataset imdb:
> python test.py --n_class 10 --dataset imdb --checkpoint ../checkpoints/imdb/timestamp

# Cite
if you use the code, please cite the following paper:

[Wu et al., 2018]  Zhen Wu, Xin-Yu Dai, Cunyan Yin, Shujian Huang, Jiajun Chen. Improving Review Representations with User Attention and Product Attention for Sentiment Classification. In Proceedings of AAAI.

# Reference
[Wu et al., 2018]  Zhen Wu, Xin-Yu Dai, Cunyan Yin, Shujian Huang, Jiajun Chen. Improving Review Representations with User Attention and Product Attention for Sentiment Classification. In Proceedings of AAAI.

[Tang et al., 2015] Duyu Tang, Bing Qin, Ting Liu. Learning Semantic Representations of Users and Products for Document Level Sentiment Classification. In Proceedings of EMNLP.
