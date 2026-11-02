"""PETR's decoder: plain multi-head attention over all camera tokens at once.

Geometry never enters the attention itself. It arrives as a 3D positional embedding added to the
keys, so a query attending to a pixel already knows where in space that pixel could be looking.
That is the whole of PETR's idea, and it is why the decoder below is an ordinary transformer.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp


class PETRMultiheadAttention(nn.Module):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding is also passed as input.
    """
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 batch_first=False,
                 **kwargs):
        super(PETRMultiheadAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first
        self.attn = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=attn_drop, **kwargs
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = nn.Identity()  # Can be replaced if needed

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                if query_pos.shape == key.shape:
                    key_pos = query_pos

        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class PETRTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 dropout=0.1,
                 with_cp=True,
                 **kwargs):
        super(PETRTransformerDecoderLayer, self).__init__()
        self.use_checkpoint = with_cp

        self.self_attn = PETRMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )
        self.norm1 = nn.LayerNorm(embed_dims)

        self.cross_attn = PETRMultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )
        self.norm2 = nn.LayerNorm(embed_dims)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(embed_dims)

    def _forward(self,
                 query,
                 key=None,
                 value=None,
                 query_pos=None,
                 key_pos=None,
                 attn_masks=None,
                 query_key_padding_mask=None,
                 key_padding_mask=None,
                 **kwargs):
        """Actual forward computation."""
        # Self attention
        temp = self.self_attn(
            query=query,
            key=query,
            value=query,
            identity=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=None
        )
        query = self.norm1(temp)

        # Cross Attention
        temp = self.cross_attn(
            query=query,
            key=key,
            value=value,
            identity=query,
            query_pos=query_pos,
            key_pos=key_pos,
            key_padding_mask=key_padding_mask,
        )
        query = self.norm2(temp)

        # FFN
        query = self.norm3(query + self.ffn(query))
        return query

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward with optional gradient checkpointing."""
        if self.use_checkpoint and self.training:
            return cp.checkpoint(
                self._forward,
                query,
                key,
                value,
                query_pos,
                key_pos,
                attn_masks,
                query_key_padding_mask,
                key_padding_mask,
                use_reentrant=False
            )
        else:
            return self._forward(
                query, key, value, query_pos, key_pos,
                attn_masks, query_key_padding_mask, key_padding_mask,
            )


class PETRTransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 dropout=0.1,
                 return_intermediate=True,
                 with_cp=True,
                 **kwargs):
        super(PETRTransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.return_intermediate = return_intermediate
        self.layers = nn.ModuleList([
            PETRTransformerDecoderLayer(
                embed_dims=embed_dims,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                dropout=dropout,
                with_cp=with_cp,
            ) for _ in range(num_layers)
        ])
        self.post_norm = nn.LayerNorm(embed_dims)

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        key_padding_mask=None,
    ):
        intermediate = []
        for layer in self.layers:
            query = layer(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
            )

            if self.return_intermediate:
                intermediate.append(self.post_norm(query))

        if self.return_intermediate:
            return torch.stack(intermediate)

        return self.post_norm(query).unsqueeze(0)


class PETRTransformer(nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_layers=6,
                 num_heads=8,
                 feedforward_channels=2048,
                 dropout=0.1,
                 with_cp=True,
                 **kwargs):
        super(PETRTransformer, self).__init__()
        self.embed_dims = embed_dims
        self.decoder = PETRTransformerDecoder(
            num_layers=num_layers,
            embed_dims=embed_dims,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            dropout=dropout,
            return_intermediate=True,
            with_cp=with_cp,
        )

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask, query_embed, pos_embed, reg_branch=None):
        """
        Args:
            x (Tensor): Input features [bs, n, c, h, w] -> memory
            mask (Tensor): key_padding_mask [bs, n, h, w]
            query_embed (Tensor): [num_query, c]
            pos_embed (Tensor): [bs, n, c, h, w]
        """
        bs, n, c, h, w = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)  # [n*h*w, bs, c]

        if pos_embed is not None:
            # [n*h*w, bs, c]
            pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c)
        else:
            pos_embed = None

        # [num_query, bs, dim]
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.reshape(bs, -1)  # [bs, n*h*w]

        target = query_embed.new_zeros(query_embed.shape)

        out_dec = self.decoder(
            query=target,
            key=memory,
            value=memory,
            key_pos=pos_embed,
            query_pos=query_embed,
            key_padding_mask=mask,
        )

        # out_dec: [num_layers, num_query, bs, dim]
        # -> [num_layers, bs, num_query, dim]
        out_dec = out_dec.transpose(1, 2).contiguous()

        return out_dec, None
