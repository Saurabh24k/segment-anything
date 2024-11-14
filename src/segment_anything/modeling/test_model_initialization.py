# test_model_initialization.py

import torch
from segment_anything.modeling import Sam, ImageEncoderViT, PromptEncoder, MaskDecoder, TwoWayTransformer

# Define model parameters
image_encoder = ImageEncoderViT(
    img_size=1024,
    patch_size=16,
    in_chans=3,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    out_chans=256,
    qkv_bias=True,
    norm_layer=nn.LayerNorm,
    act_layer=nn.GELU,
    use_abs_pos=True,
    use_rel_pos=False,
    rel_pos_zero_init=True,
    window_size=0,
    global_attn_indexes=(),
)

prompt_encoder = PromptEncoder(
    embed_dim=256,
    image_embedding_size=(64, 64),  # Adjust based on your image encoder output
    input_image_size=(1024, 1024),
    mask_in_chans=16,
    activation=nn.GELU,
    num_attention_heads=8,
)

transformer = TwoWayTransformer(  # Ensure you have this module
    depth=2,
    embedding_dim=256,
    num_heads=8,
)

mask_decoder = MaskDecoder(
    transformer_dim=256,
    transformer=transformer,
    num_multimask_outputs=3,
    activation=nn.GELU,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
    num_attention_heads=8,
)

model = Sam(
    image_encoder=image_encoder,
    prompt_encoder=prompt_encoder,
    mask_decoder=mask_decoder,
)

print("Model initialized successfully!")
