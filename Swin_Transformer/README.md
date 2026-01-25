# Swin Transformer (Tiny) PyTorch Implementation

An implementation of the Swin Transformer in PyTorch, trained on Tiny ImageNet.
This project is based on the original paper "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", with several modifications from the published design. Swin Transformers are especially effective on large scale datasets with high resolution inputs, where ViT's quadratic self-attention complexity is not ideal. Tiny ImageNet's 64x64 resolution images are tiny, with a 4x4 patch size, the feature map is reduced to 16x16 tokens in the first stage, after three rounds, it is operating on a 2x2 token grid, leaving little spatial detail for the classifier. Validation Top1 accuracy plateaued around 56% which is around what I was expecting. While Tiny ImageNet is useful for experimentation, training on the full ImageNet would better demonstrate Swin's scalability and performance advantages.

- **Input size**: 64×64 RGB
- **Augmentations**
  - RandomResizedCrop
  - RandomHorizontalFlip
  - Normalization
  - ColorJitter
  - RandAugment
  - Random Erasing
  - Stochastic Depth
  - Mixup
  - CutMix

## Parameters

- Patch size: 4 × 4
- Base embed dim C: 96
- Depths: [2, 2, 6, 2]
- Num heads: [2, 4, 8, 16]
- Window size: 7 for all blocks
- Shift: 3 (7 // 2) for all shift blocks
- MLP expantion ratio: 4.0 for all blocks
- Drop path rate: 0.2 (linearly increased across all blocks)
- Patch Merging / Downsample at the end of Stage 1, 2 and 3.


| Stage         | Blocks | Heads | Stoch_dep | In Channels | Out Channels | Output Shape             |
|---------------|--------|-------|-----------|-------------|--------------|--------------------------|
| PatchEmbed    | None   | None  | None           | 3           | 96           | (BS, H/4, W/4, 96)       |
| Stage 1       | 2      | 2     | [0.0000, 0.0182]          | 96          | 192          | (BS, H/8, W/8, 192)      |
| Stage 2       | 2      | 4     | [0.0364, 0.0545]          | 192         | 384          | (BS, H/16, W/16, 384)    |
| Stage 3       | 6      | 8     | [0.0727, 0.0909, 0.1091, 0.1273, 0.1455, 0.1636]          | 384         | 768          | (BS, H/32, W/32, 768)    |
| Stage 4       | 2      | 16    | [0.1818, 0.2000]          | 768         | 768          | (BS, H/32, W/32, 768)    |
| Head          | None   | None  | None          | 768         | num_classes  | (BS, num_classes)        |

## Swin Transformer Architecture

![Architecture](figures/Architecture.png)

## SwinBlock (Shifted Window Transformer Block)

Applies windowed self-attention over non overlapping M×M windows and alternates between non-shifted and shifted windows across consecutive blocks to enable cross-window connections. There is also an MLP with GELU at the end. Residual connections are used.

![Block Architecture](figures/Block_arch.png)

## Constraints

- img_size must be divisible by patch_size * 2^3
- For each stage, H and W should be multiples of the window size M

## Modules

- **Patchify:** Conv2d patch embedding (BS, 3, H, W) --> (BS, H/4, W/4, channels)
- **WindowAttention:** Multi-head attention over M×M tokens with relative position bias and an optional attention mask
- **create_attention_mask:** Prevents cross-window attention after cyclic shift
- **StochasticDepth:** Drops residual branches
- **PatchMerging:** Downsample 2x spatially, project channels --> 2*channels
- **Stage:** Stacks SwinBlocks with alternating shifts. Optional PatchMerging at the end

## Training

- **Optimizer:** AdamW(param_groups_weight_decay(model, 0.05), lr=3e-4, betas=(0.9, 0.999), fused=True)
- **LR schedule:** Linear warmup for 10 epochs, then Cosine for 190 epochs
- **Precision:** AMP (torch.cuda.amp Autocast + GradScaler)
- **Regularization:** Gradient clipping (max_norm=3.0), stochastic depth
- **Logging:** train_loss, train_acc, val_loss, val_acc, lr --> [2, 2, 4, 2].json, [2, 2, 6, 2].json

![6blockAcc](figures/6blockAcc.png)

## Notes

Notes I gathered while researching. They are not organized. Kept here so I can easily access them.

Swin Transformer does self-attention in patches instead of self attention globally. It has linear computation complexity to input image size due to computation of self-attention only within each local window. One problem is there is a much higher resolution of pixels in images compared to words in passages of text. Semantic Segmentation is assigning a class to every single pixel in an image. It would be intractable for a Transformer on high-resolution images due to the complexity of self-attention being quadratic to image size. The Swin Transformer overcomes this issue. It constructs hierarchical feature maps and has linear computational complexity to image size. It starts with small sized patches and gradually merges those patches together in deeper Transformer layers. With these hierarchical feature maps, the Swin Transformer can use advanced techniques for dense prediction such as Feature Pyramid Networks (FPN) Which detect objects at different scales effectively. Small objects appear in high resolution regions (early layers of CNNs), large objects appear in low resolution deep layers (later CNN layers). Deep CNNs tend to lose spatial resolution as they go deeper, shallow layers have high resolution but low semantic richness. FPN combines high resolution spatial info from shallow layers with high level semantic info from deep layers. It builds a multi-scale feature pyramid from a single input image by using top-down pathways and lateral connections on top of a CNN backbone like ResNet. Bottom-Up Pathway is a standard CNN that outputs feature maps at different scales. Top-Down Pathway starts from the deepest layer and progressively upsample feature maps. At each step you combine the unsampled feature with a shallower feature map from the bottom-up pathway via Lateral Connections: These are 1x1 convolutions that project the bottom-up features to the same channel size as the top-down features. You add them element-wise to the upsampled feature map. P5 = conv1x1(C5), 
P4 = upsample(P5) + conv1x1(C4), P3 = upsample(P4) + conv1x1(C3). No point writing down how to upsample in notes. The other technique is U-Net which is a CNN designed for image segmentation. It’s one of the most widely used models for pixel-wise prediction tasks. It’s built around the idea of encoding the context of the image and decoding it back to the original resolution to make dense, pixel-level predictions. The encoder is a series of conv → ReLU → conv → ReLU → maxpool blocks. Each block halves the spatial resolution and doubles the number of channels. It captures what is in the image, semantic information. The decoder is a series of upsampling → convolution → ReLU → concatenation with encoder features. Spatial resolution is doubled at each step. It combines context from the encoder (via skip connections) with precise localization. The skip connections copy and concatenate features from the encoder to the decoder at each resolution level. It helps retain fine grained spatial information that is lost in downsampling. The original U-Net doesn’t scale well to very deep or large datasets. There are many variants (U-Net++, Attention U-Net, TransUNet). A key design element of Swin Transformer is its shift of the window partition between consecutive self-attention layers. Since self-attention is computed in windows, tokens in different windows never interact. Shifted windows bridge the windows of the preceding layer, providing connections among them that significantly enhance modeling power. All query patches within a window share the same key set which facilitates memory access in hardware. Swin Transformer outperforms ViT / DeiT(Distillation token one) and ResNeXt models significantly with similar latency on image classification, object detection, and semantic segmentation. It achieves a top-1 accuracy of 87.3% on ImageNet-1K. Swin Transformers split the input RGB image into non overlapping patches like ViT. Each patch is treated as a token and its feature is a concatenation of the raw pixel RGB values. Swin uses a patch size of 4x4 and thus the feature dimension of each patch is 4 x 4 x 3 = 48. A linear embedding layer is applied on this raw valued feature to project it into an arbitrary dimension. Several Transformer blocks with modified self-attention computation (Swin Transformer Blocks) are applied on these patch tokens. The Transformer blocks maintain the number of tokens (H/4 x W/4) and together with the linear embedding are referred to as Stage 1. To produce a hierarchical representation. The number of tokens is reduced by patch merging layers as the network gets deeper. The first patch merging layer concatenates the features of each group of 2x2 neighboring patches. And applies a linear layer on the 4\*d_model dimensional concatenated features. This reduces the number of tokens by 4 (2x downsampling) And the output dimension is set to 2\*d_model. Swin Transformer blocks are applied afterwards for feature transformation with the resolution kept at H/8 x W/8. This first block of patch merging and feature transformation is denoted as Stage 2. The procedure is repeated twice as Stage 3 and Stage 4 with output resolutions of H/16 x W/16 and H/32 x W/32.

Swin Transformer Block is built by replacing the standard multi-head self attention (MSA) module in a Transformer block by a module based on shifted windows. A Swin Transformer block consists of a shifted window based MSA module followed by a 2-layer MLP with GELU. A LayerNorm (LN) layer is applied before each MSA module and each MLP and a residual connection is applied after each module. zˆl = W-MSA(LN(zl−1))  + zl−1,  zl = MLP(LN(zˆ l)  + zˆ l) Where zˆl denotes the output features of (S)W-MSA module for block l and zl denotes the output features of the MLP module for block l. Simple formula with residual connection at the end. MLP(x) = Linear(x) → GELU → Dropout → Linear → Dropout. When you shift you would need to add more windows and sizes will change. Solution is Cyclic Shifting: swap vertical and horizontal. x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2)) this moves each row up shift_size pixels, and each column left by shift_size pixels. It wraps around at edges. Shift size is normally half of the window size. You need to use a mask to ensure attention stays within each window. After attention, shift back. After the cyclic shift each shifted window contains tokens from different original windows. The attention mask is used so that tokens only attend to others in their current window. Swin Transformer does not use absolute position embeddings like the ViT, instead it uses Relative Position Bias within each window. 6x Swin Transformer blocks would look like this: W-MSA → SW-MSA → W-MSA → SW-MSA → W-MSA → SW-MSA For each SW-MSA, you shift and then reset. Details of ImageNet-1k Swin Transformer:  224x224 input, employ an AdamW optimizer for 300 epochs using a cosine decay learning rate scheduler with 20 epochs of linear warm-up. A batch size of 1024, an initial learning rate of 0.001, a weight decay of 0.05, and gradient clipping with a max norm of 1 are used. Include RandAugment (Applies random augmentations like rotation, translation), Mixup (Combines two input images and their labels linearly, xnew = 入xi + (1 - 入)xj), CutMix (Replaces a patch from one image with a patch from another image), random erasing (Randomly erases a rectangular patch from an image) and stochastic depth (Randomly drops entire residual blocks during training, dropout for layers), but not repeated augmentation (Uses multiple differently augmented versions of the same image in the same batch) and Exponential Moving Average (EMA) (Keeps a moving average of model weights) which do not enhance performance. Note that this is contrary to where repeated augmentation is crucial to stabilize the training of ViT. An increasing degree of stochastic depth augmentation is employed for larger models, i.e. 0.2, 0.3, 0.5 for Swin-T, Swin-S, and Swin-B, respectively.

Built by Kaizen Rowe