import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalAutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.image_token_embed = nn.Embedding(vocab_size, embed_dim)
        self.text_condition_proj = nn.Linear(768, embed_dim)
        self.pos_embedding = nn.Embedding(1024, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=1024
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)

    def forward(self, image_tokens, text_embedding):
        if image_tokens.dim() == 3:
            batch_size, height, width = image_tokens.shape
            seq_len = height * width
            image_tokens = image_tokens.view(batch_size, seq_len)
        elif image_tokens.dim() == 2:
            batch_size, seq_len = image_tokens.shape
        else:
            raise ValueError(f"Unexpected shape for image_tokens: {image_tokens.shape}")

        image_token_embeds = self.image_token_embed(image_tokens)
        positions = torch.arange(seq_len, device=image_tokens.device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.pos_embedding(positions)
        image_token_embeds = image_token_embeds + pos_embeds
        
        text_embeds = self.text_condition_proj(text_embedding).unsqueeze(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(image_tokens.device)
        
        output = self.transformer_decoder(
            tgt=image_token_embeds.permute(1, 0, 2),
            memory=text_embeds.permute(1, 0, 2),
            tgt_mask=causal_mask
        )
        
        output = self.output_layer(output.permute(1, 0, 2))
        return output