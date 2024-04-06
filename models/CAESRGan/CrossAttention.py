import torch 
import torch.nn as nn

class CrossAttention(nn.Module):
    def __init__(self, len_seq1, len_seq2, in_channel):
        super().__init__()
        self.len_seq1 = len_seq1
        self.len_seq2 = len_seq2
        self.in_channel = in_channel

        # Self-attention
        self.self_attn = nn.MultiheadAttention(in_channel, num_heads=1)

        # Cross-attention between seq1 and seq2
        self.cross_attn = nn.MultiheadAttention(in_channel, num_heads=1)

        # Fully connected layer for final output
        self.fc = nn.Linear(in_channel * 2, in_channel)

        # Positional Embedding 
        self.pos_embedding1 = nn.Parameter(torch.randn(in_channel, self.len_seq1))
        self.pos_embedding2 = nn.Parameter(torch.randn(in_channel, self.len_seq2))

    def forward(self, img1, img2):
        batch_size = img1.size(0)

        # Flatten the resolution  
        img1_flat = img1.view(batch_size, img1.size(1), -1)  # (batch_size, channels, height*width)
        img2_flat = img2.view(batch_size, img2.size(1), -1)  # (batch_size, channels, height*width)

        # Add positional embedding
        img1_with_pos = img1_flat + self.pos_embedding1.unsqueeze(0)
        img2_with_pos = img2_flat + self.pos_embedding2.unsqueeze(0)

        # Reshape images for self-attention
        img1_reshaped = img1_with_pos.permute(2, 0, 1)  # (seq_len, batch_size, in_channel)
        img2_reshaped = img2_with_pos.permute(2, 0, 1)  # (seq_len, batch_size, in_channel)

        # Self-Attention on img1
        self_attn_output, _ = self.self_attn(img1_reshaped, img1_reshaped, img1_reshaped)
        self_attn_output = self_attn_output.permute(1, 0, 2)  # (batch_size, seq_len, in_channel)

        # Cross-Attention from img2 to img1
        cross_attn_output, _ = self.cross_attn(img1_reshaped, img2_reshaped, img2_reshaped)
        cross_attn_output = cross_attn_output.permute(1, 0, 2)  # (batch_size, seq_len, in_channel)

        # Concatenate self-attention output and cross-attention output
        combined_output = torch.cat((self_attn_output, cross_attn_output), dim=-1)

        # Apply fully connected layer
        output = self.fc(combined_output)

        # Reshape the output
        output = output.view(img1.size(0), img1.size(1), img1.size(2), img1.size(3))

        return output 
