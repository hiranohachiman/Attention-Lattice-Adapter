import torch
import torch.nn as nn
import math
class TransformerDecoder(nn.Module):
    def __init__(self,  num_hidden_layers=5):
        super().__init__()
        self.layer = nn.ModuleList(
            [DecoderLayer() for _ in range(num_hidden_layers)]
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        ):

        for layer in self.layer:
            hidden_states = layer(q, kv)

        return hidden_states

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(512, eps=1e-12)

        self.attention = MultiHeadAttention()
        self.attention2 = MultiHeadAttention()
        self.attention3 = MultiHeadAttention()
        # ffn
        self.ffn = FeedforwardNeuralNetModel(512, 1024, 512)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        ):

        # attention layer q=text, k,v=image
        identity_x = q
        att = self.attention(x=q, source_kv=kv)
        q = identity_x + att
        q = self.LayerNorm(q)
        identity_x = q
        q = self.ffn(q)
        output = identity_x + q
        output = self.LayerNorm(output)

        return output



class MultiHeadAttention(nn.Module):
    """
    TransformerにおけるMHA
    """

    def __init__(self, hidden_size=512, num_attention_heads=8):
        super().__init__()
        self.self = SelfAttention()
        self.output = SelfOutput()
        self.layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None, source_kv=None):
        """
        Args:
            input_tensor: (N, L, D)
            attention_mask: (N, Lq, L)
        Returns:
        """
        if source_kv is not None:
            # self_output = self.self(source_kv, x, x, attention_mask) #(16,63,768)
            self_output = self.self(x, source_kv, source_kv, attention_mask)
            # att = self.output(self.output, x)
        else:
            self_output = self.self(x, x, x, attention_mask)
        att = self.output(self_output, x)
        return att

class SelfAttention(nn.Module):
    """
    Attentionの計算
    """

    def __init__(self, hidden_size=512, num_attention_heads=8):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_w = nn.Linear(hidden_size, self.all_head_size)
        self.key_w = nn.Linear(hidden_size, self.all_head_size)
        self.value_w = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(0.1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )  # (N, L, nh, dh)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (N, nh, L, dh)

    def forward(self, query, key, value, attention_mask=None):
        """
        Args:
            query_states: (N, Lq, D)
            key_states: (N, L, D)
            value_states: (N, L, D)
            attention_mask: (N, Lq, L)

        Returns:
        """
        # only need to mask the dimension where the softmax
        # (last dim) is applied, as another dim (second last)
        # will be ignored in future computation anyway
        if attention_mask is not None:
            attention_mask = \
                (1 - attention_mask.unsqueeze(1)) * -10000.0  # (N, 1, Lq, L)

        mixed_query_layer = self.query_w(query)
        mixed_key_layer = self.key_w(key)
        mixed_value_layer = self.value_w(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)  # (N, nh, Lq, dh)
        key_layer = self.transpose_for_scores(mixed_key_layer)  # (N, nh, L, dh)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # (N, nh, L, dh)

        # calc attention
        att_w = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (N, nh, Lq, L)
        att_w = att_w / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers
            # in BertModel forward() function)
            att_w = att_w + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(att_w)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class SelfOutput(nn.Module):
    """
    TransformerにおけるFF層
    """

    def __init__(self,  hidden_size=512, num_attention_heads=12):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)

        return out
