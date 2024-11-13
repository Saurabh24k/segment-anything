# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d
import torch.nn.functional as F
from torch.nn import MultiheadAttention


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
        num_attention_heads: int = 8,
        num_hierarchy_levels: int = 2,  # Define the number of hierarchy levels
    ) -> None:
        super().__init__()

        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """

        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.point_embeddings = nn.ModuleDict({
            'positive': nn.Embedding(1, embed_dim),
            'negative': nn.Embedding(1, embed_dim),
        })
        self.box_embeddings = nn.ModuleDict({
            'box_corner1': nn.Embedding(1, embed_dim),
            'box_corner2': nn.Embedding(1, embed_dim),
        })
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

        # Multi-Headed Self-Attention Layer
        self.attention_layer = MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            batch_first=True  # To keep batch dimension first
        )

        # Hierarchical Encoding Layers
        self.activation = activation()
        self.num_hierarchy_levels = num_hierarchy_levels
        self.hierarchical_layers = nn.ModuleList([
            nn.Linear(self.embed_dim, self.embed_dim) for _ in range(self.num_hierarchy_levels)
        ])
        
    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        # Replace direct weight access with embedding lookup
        negative_embedding = self.point_embeddings['negative'](torch.zeros_like(labels, dtype=torch.long))
        positive_embedding = self.point_embeddings['positive'](torch.zeros_like(labels, dtype=torch.long))
        not_a_point_embedding = self.not_a_point_embed(torch.zeros_like(labels, dtype=torch.long))

        # Apply embeddings based on labels
        point_embedding = torch.where(
            labels.unsqueeze(-1) == -1,
            point_embedding + not_a_point_embedding,
            point_embedding
        )
        point_embedding = torch.where(
            labels.unsqueeze(-1) == 0,
            point_embedding + negative_embedding,
            point_embedding
        )
        point_embedding = torch.where(
            labels.unsqueeze(-1) == 1,
            point_embedding + positive_embedding,
            point_embedding
        )

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_indices = torch.zeros((coords.shape[0], 2), dtype=torch.long, device=boxes.device)
        corner_indices[:, 1] = 1  # Second corner
        box_corner_embeddings = self.box_embeddings['box_corner1'](corner_indices[:, 0]) + \
                                self.box_embeddings['box_corner2'](corner_indices[:, 1])
        corner_embedding += box_corner_embeddings.unsqueeze(1)
        return corner_embedding


    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings_list = []

        for level in range(self.num_hierarchy_levels):
            scale_factor = 1 / (2 ** level)
            level_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())

            if points is not None:
                coords, labels = points
                scaled_coords = coords * scale_factor
                point_embeddings = self._embed_points(scaled_coords, labels, pad=(boxes is None))
                level_embeddings = torch.cat([level_embeddings, point_embeddings], dim=1)

            if boxes is not None:
                scaled_boxes = boxes * scale_factor
                box_embeddings = self._embed_boxes(scaled_boxes)
                level_embeddings = torch.cat([level_embeddings, box_embeddings], dim=1)

            sparse_embeddings_list.append(level_embeddings)

        # Concatenate embeddings from all levels
        sparse_embeddings = torch.cat(sparse_embeddings_list, dim=1)

        # Apply Self-Attention to Sparse Embeddings
        if sparse_embeddings.size(1) > 0:
            # Apply hierarchical encoding
            for layer in self.hierarchical_layers:
                sparse_embeddings = layer(sparse_embeddings)
                sparse_embeddings = self.activation(sparse_embeddings)
            # Apply Self-Attention to Sparse Embeddings
            if sparse_embeddings.size(1) > 1:
                attn_output, _ = self.attention_layer(
                    sparse_embeddings,  # Query
                    sparse_embeddings,  # Key
                    sparse_embeddings   # Value
                )
                sparse_embeddings = sparse_embeddings + attn_output  # Residual Connection

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings



class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )


    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C
