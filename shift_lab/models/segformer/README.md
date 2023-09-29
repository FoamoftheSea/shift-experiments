# Multitask Segformer

https://github.com/FoamoftheSea/shift-experiments/assets/50897218/0b43a121-a1e6-43ac-b80e-39842e2c225d

Segformer ([Xie et al., 2021](https://arxiv.org/abs/2105.15203)) showed that a lightweight hierarchical transformer encoder could deliver multi-scale features to an all-MLP decoding head to deliver powerful semantic segmentation results. The next year, GLPN ([Kim et al., 2022](https://arxiv.org/abs/2201.07436)) showed that the same encoder could be paired with a lightweight decoding head to achieve competitive monocular depth estimation performance.

This model is a combination of the Segformer and GLPN models, which jointly estimates semantic segmentation and depth using the features from a single Segformer B0-B5 encoder passed into the separate task heads from Segformer and GLPN. Training can be turned off or on separately for the task heads, and the model may optionally be used for single tasks.

With a B0 backbone and only 4.07M parameters, the model shows strong performance on the synthetic SHIFT dataset. To see the model in action, refer to the [blog post](https://hiddenlayers.tech/blog/segformer-demonstrates-powerful-multitask-performance). For a detailed breakdown of the training method and results, see the [Weights and Biases report](https://api.wandb.ai/links/indezera/4ua2bsyk).

For setup instructions, see the project [README](../../../README.md).

```
python train_segformer.py --help
```
