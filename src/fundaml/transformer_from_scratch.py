import torch
import torch.nn as nn
from torch.autograd import Variable

import unittest

def get_device():
    # Check CUDA availability
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')    

class MultiHeadSelfAttentionBlock(nn.Module):
    """
    Initializes the MultiHeadSelfAttentionBlock module.

    Args:
        embedding_size (int): The size of the input embeddings. The value should be 
                            divisible by num_heads. This is because the embeddings 
                            are split into `num_heads` different pieces during the 
                            self-attention process.
        num_heads (int): The number of attention heads. In the multi-head attention 
                        mechanism, the model generates `num_heads` different attention 
                        scores for each token in the sequence. This allows the model 
                        to focus on different parts of the sequence for each token.

    Raises:
        AssertionError: If `embedding_size` is not divisible by `num_heads`. The embedding 
                        size needs to be divisible by the number of heads to ensure even 
                        division of embeddings for multi-head attention.
    """    
    def __init__(self, embedding_size, num_heads,device=None):
        super(MultiHeadSelfAttentionBlock, self).__init__()
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        assert (
            self.head_dim * num_heads == embedding_size
        ), "embedding_size  needs to be divisible by num_heads"

        # Define the linear transformations for the input data 
        self.values_transform = nn.Linear(embedding_size, embedding_size)
        self.keys_transform = nn.Linear(embedding_size, embedding_size)
        self.queries_transform = nn.Linear(embedding_size, embedding_size)
        # Define the final output linear transformation
        self.linear_out = nn.Linear(embedding_size, embedding_size)

    def forward(self, values, keys, queries, mask):
        """
        Forward pass for the MultiHeadSelfAttentionBlock module.

        Args:
            values (torch.Tensor): The values tensor of shape (N, value_len, embedding_size),
                where N is the batch size, value_len is the sequence length for the values.
            keys (torch.Tensor): The keys tensor of shape (N, key_len, embedding_size), 
                where key_len is the sequence length for the keys.
            queries (torch.Tensor): The query tensor of shape (N, query_len, embedding_size), 
                where query_len is the sequence length for the queries.
            mask (torch.Tensor, optional): A mask tensor of shape (N, 1, 1, query_len/key_len),
                where the values are either 1 (for positions to be attended to) or 0 
                (for positions to be masked). The mask is used to prevent attention to 
                certain positions. Default is None.

        Returns:
            torch.Tensor: The output tensor of shape (N, query_len, embedding_size), 
                where N is the batch size, query_len is the sequence length for the queries,
                and embedding_size is the dimension of the output embeddings. 
                This tensor represents the result of applying self-attention on the input.

        Note:
            The method first transforms the input tensors (values, keys, query) using 
            separate linear transformations. Then, it splits the transformed embeddings 
            into multiple heads and computes the attention scores (queries * keys). If a mask 
            is provided, it is applied to the scores. The scores are then normalized to create 
            the attention weights. The method then computes the weighted sum of the values 
            (attention * values), applies a final linear transformation, and returns the result.
        """        
        # Get the number of training examples
        num_examples = queries.shape[0]

        # Calculate sequence lengths
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        
        values = values.to(self.device)
        keys = keys.to(self.device)
        queries = queries.to(self.device)
        mask = mask.to(self.device) if mask is not None else None

        # Transform the input data
        values = self.values_transform(values)
        keys = self.keys_transform(keys)
        queries = self.queries_transform(queries)

        # Split the embeddings into multiple heads
        values = values.reshape(num_examples, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(num_examples, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(num_examples, query_len, self.num_heads, self.head_dim)

        # Compute attention scores (queries * keys)
        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, num_heads, heads_dim),
        # keys shape: (N, key_len, num_heads, heads_dim)
        # attention_scores: (N, num_heads, query_len, key_len)        

        # Apply the mask to the scores
        # mask shape: (N, 1, 1, query_len/key_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))

        # Normalize the scores
        # attention shape: (N, num_heads, query_len, key_len)
        attention = torch.softmax(attention_scores / (self.embedding_size ** 0.5), dim=3)

        # Compute weighted values (attention * values)
        # attention shape: (N, num_heads, query_len, key_len)
        # values shape: (N, value_len, num_heads, heads_dim)
        # weighted_values shape: (N, query_len, num_heads, heads_dim)
        weighted_values = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        weighted_values = weighted_values.reshape(num_examples, query_len, self.num_heads * self.head_dim)

        # Apply the final linear transformation
        out = self.linear_out(weighted_values)

        return out

class TransformerBlock(nn.Module):
    """
    A Transformer Block class that defines a single block in a transformer model.

    Args:
        embed_size (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads for the self-attention mechanism.
        dropout_rate (float): The dropout rate used in the dropout layers to prevent overfitting.
        forward_expansion (int): The factor by which the dimensionality of the input is 
            expanded in the feed-forward network. The feed-forward network expands the 
            dimensionality of the input from `embed_size` to `forward_expansion * embed_size`
            and then reduces it back to `embed_size`.

    Attributes:
        self_attention (SelfAttention): The self-attention layer used in the transformer block.
        norm1 (nn.LayerNorm): The first layer normalization used to stabilize the outputs of 
            the self-attention layer.
        norm2 (nn.LayerNorm): The second layer normalization used to stabilize the outputs 
            of the feed-forward network.
        feed_forward (nn.Sequential): The feed-forward network used to transform the output 
            of the self-attention layer.
        dropout (nn.Dropout): The dropout layer used to prevent overfitting.

    The Transformer Block consists of a self-attention layer followed by normalization, 
    a feed-forward network followed by normalization, and a dropout layer. 
    """    
    def __init__(self, embed_size, num_heads, dropout_rate, forward_expansion, device=None):
        super(TransformerBlock, self).__init__()
        if device is None:
            self.device = get_device()
        else:
            self.device = device
        # Transformer block consists of a self-attention layer and a feed-forward neural network
        self.self_attention = MultiHeadSelfAttentionBlock(embed_size, num_heads,device=device)
        
        # Normalization layers are used to stabilize the outputs of self-attention and feed-forward network
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        # Feed-forward neural network - it's used to transform the output of the self-attention layer
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        # Dropout is used to reduce overfitting
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, values, keys, queries, mask):
        """
        Forward pass of the Transformer Block.

        Args:
            values (torch.Tensor): The values used by the self-attention layer. 
                They have shape (N, value_len, embed_size) where N is the batch size, 
                value_len is the length of the value sequence, and embed_size is the size of the embeddings.
            keys (torch.Tensor): The keys used by the self-attention layer. 
                They have shape (N, key_len, embed_size) where N is the batch size, 
                key_len is the length of the key sequence, and embed_size is the size of the embeddings.
            queries (torch.Tensor): The queries used by the self-attention layer. 
                They have shape (N, query_len, embed_size) where N is the batch size, 
                query_len is the length of the query sequence, and embed_size is the size of the embeddings.
            mask (torch.Tensor): The mask to be applied on the attention outputs to prevent the model 
                from attending to certain positions. It has shape (N, 1, 1, src_len), 
                where N is the batch size and src_len is the source sequence length.

        Returns:
            out (torch.Tensor): The output tensor from the transformer block, it has shape 
                (N, query_len, embed_size), where N is the batch size, query_len is the length 
                of the query sequence, and embed_size is the size of the embeddings.

        The forward method first applies the self-attention mechanism on the input tensor using the provided
        keys, queries, and values. The output from the self-attention layer is then passed through a
        normalization layer and a dropout layer. The output from these layers is then passed through the
        feed-forward network. The output from the feed-forward network is also passed through a 
        normalization layer and a dropout layer. The final output is then returned.
        """
        values = values.to(self.device)
        keys = keys.to(self.device)
        queries = queries.to(self.device)
        mask = mask.to(self.device) if mask is not None else None
        
        # Self-attention layer takes in values, keys and queries and returns an output tensor
        attention_output = self.self_attention(values, keys, queries, mask)

        # Add residual connection (skip connection), normalize and apply dropout
        # The normalization is applied on the sum of the original input `queries` and the output of the self-attention layer
        x = self.dropout(self.norm1(attention_output + queries))

        # Pass the output from the attention layer through the feed-forward network
        ff_output = self.feed_forward(x)
        
        # Add another residual connection, normalize and apply dropout
        # The normalization is applied on the sum of the previous output `x` and the output of the feed-forward network
        out = self.dropout(self.norm2(ff_output + x))

        return out

class Encoder(nn.Module):
    """
    The Encoder class for a Transformer model.
    
    Args:
        src_vocab_size (int): The size of the source vocabulary.
        embed_size (int): The dimensionality of the input embeddings.
        num_layers (int): The number of layers in the transformer.
        num_heads (int): The number of attention heads in the transformer block.
        device (torch.device): The device to run the model on (CPU or GPU).
        forward_expansion (int): The expansion factor for the feed forward network in transformer block.
        dropout (float): The dropout rate used in the dropout layers to prevent overfitting.
        max_length (int): The maximum sequence length the model can handle.
    """
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        num_heads,
        forward_expansion,
        dropout_rate,
        max_length,
        device=None,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        # Embeddings for the input words and positional embeddings
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Transformer blocks for the encoder layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    num_heads,
                    dropout_rate=dropout_rate,
                    forward_expansion=forward_expansion,
                    device=self.device,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        """
        Forward method for the Encoder class.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length).
            mask (torch.Tensor): The mask to be applied on the attention outputs to prevent the model 
                from attending to certain positions.
        
        Returns:
            out (torch.Tensor): The output tensor from the encoder.
        """

        # Obtain the batch size and sequence length
        N, seq_length = x.shape
        x = x.to(self.device)
        mask = mask.to(self.device)

        # Create positional indices
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # Combine word embeddings and positional embeddings
        embeddings = self.word_embedding(x) + self.position_embedding(positions)

        # Apply dropout to the combined embeddings
        out = self.dropout(embeddings)

        # Pass the output through the layers
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    """
    The DecoderBlock class that forms a part of the Decoder in a Transformer model.

    Args:
        embed_size (int): The dimensionality of the input embeddings.
        num_heads (int): The number of attention heads for the self-attention mechanism.
        forward_expansion (int): The factor by which the dimensionality of the input is expanded in the feed-forward network.
        dropout_rate (float): The dropout rate used in the dropout layers to prevent overfitting.
        device (torch.device): The device to run the model on (CPU or GPU).
        
    Attributes:
        norm (nn.LayerNorm): Layer normalization.
        attention (SelfAttention): The self-attention mechanism.
        transformer_block (TransformerBlock): A transformer block.
        dropout (nn.Dropout): Dropout layer for regularization.
    """
    def __init__(self, embed_size, num_heads, forward_expansion, dropout_rate, device=None):
        super(DecoderBlock, self).__init__()
        if device is None:
            self.device = get_device()
        else:
            self.device = device        
        self.norm = nn.LayerNorm(embed_size)
        self.self_attention = MultiHeadSelfAttentionBlock(embed_size, num_heads)
        self.transformer_block = TransformerBlock(
            embed_size, num_heads, dropout_rate, forward_expansion
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, value, key, src_mask, trg_mask):
        """
        Forward method for the DecoderBlock class.
        
        Args:
            x (torch.Tensor): The input tensor.
            value (torch.Tensor): The values to be used in the self-attention mechanism.
            key (torch.Tensor): The keys to be used in the self-attention mechanism.
            src_mask (torch.Tensor): The source mask to prevent attention to certain positions.
            trg_mask (torch.Tensor): The target mask to prevent attention to certain positions.
        
        Returns:
            out (torch.Tensor): The output tensor from the transformer block.
        """

        x = x.to(self.device)
        value = value.to(self.device)
        key = key.to(self.device)
        src_mask = src_mask.to(self.device)
        trg_mask = trg_mask.to(self.device)
        
        # Compute self attention
        attention = self.self_attention(x, x, x, trg_mask)
        
        # Apply normalization and dropout to the sum of the input and the attention output
        query = self.dropout(self.norm(attention + x))
        
        # Compute the output of the transformer block
        out = self.transformer_block(value, key, query, src_mask)
        
        return out

class Decoder(nn.Module):
    """
    The Decoder class for a Transformer model.
    
    Args:
        trg_vocab_size (int): The size of the target vocabulary.
        embed_size (int): The dimensionality of the input embeddings.
        num_layers (int): The number of layers in the transformer.
        num_heads (int): The number of attention heads in the transformer block.
        forward_expansion (int): The expansion factor for the feed forward network in transformer block.
        dropout_rate (float): The dropout rate used in the dropout layers to prevent overfitting.
        device (torch.device): The device to run the model on (CPU or GPU).
        max_length (int): The maximum sequence length the model can handle.
    """
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        num_heads,
        forward_expansion,
        dropout_rate,
        max_length,
        device = None
    ):

        super(Decoder, self).__init__()
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        # Embeddings for the input words and positional embeddings
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        # Decoder blocks for the decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_size,
                    num_heads,
                    forward_expansion,
                    dropout_rate,
                    device=self.device
                )
                for _ in range(num_layers)
            ]
        )

        # Fully connected layer to map the decoder's output to the target vocabulary size
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_out, src_mask, trg_mask):
        """
        Forward method for the Decoder class.
        
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_length).
            enc_out (torch.Tensor): The output from the encoder.
            src_mask (torch.Tensor): The source mask to prevent the model from attending to certain positions.
            trg_mask (torch.Tensor): The target mask to prevent the model from attending to certain positions.
        
        Returns:
            out (torch.Tensor): The output tensor from the decoder.
        """

        # Obtain the batch size and sequence length
        N, seq_length = x.shape
        x = x.to(self.device)
        enc_out = enc_out.to(self.device)
        src_mask = src_mask.to(self.device)
        trg_mask = trg_mask.to(self.device)

        # Create positional indices
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        # Combine word embeddings and positional embeddings
        embeddings = self.word_embedding(x) + self.position_embedding(positions)

        # Apply dropout to the combined embeddings
        x = self.dropout(embeddings)

        # Pass the output through the layers
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        # Apply the fully connected layer to the outputs of the layers
        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    """
    The Transformer model class that combines an Encoder and a Decoder.

    Args:
        src_vocab_size (int): The size of the source vocabulary.
        trg_vocab_size (int): The size of the target vocabulary.
        src_pad_idx (int): The index of the source padding token in the source vocabulary.
        trg_pad_idx (int): The index of the target padding token in the target vocabulary.
        embed_size (int): The dimensionality of the input embeddings.
        num_layers (int): The number of layers in the transformer.
        forward_expansion (int): The expansion factor for the feed forward network in transformer block.
        heads (int): The number of attention heads in the transformer block.
        dropout (float): The dropout rate used in the dropout layers to prevent overfitting.
        device (torch.device): The device to run the model on (CPU or GPU).
        max_length (int): The maximum sequence length the model can handle.
    """

    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=512,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        max_length=100,
        device=None
    ):

        super(Transformer, self).__init__()
        if device is None:
            self.device = get_device()
        else:
            self.device = device  
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device=self.device
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            max_length,
            device=self.device
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    @staticmethod
    def make_src_mask(src,src_pad_idx,device=None):
        """
        Creates a mask for the source input sequence.
        
        Args:
            src (torch.Tensor): The source input sequence.
        
        Returns:
            src_mask (torch.Tensor): The mask for the source input sequence.
        """
        if device is None:
            device = get_device()
        src_mask = (src != src_pad_idx).unsqueeze(1).unsqueeze(2).to(device)
        return src_mask

    @staticmethod
    def make_trg_mask(trg,device=None):
        """
        Creates a mask for the target input sequence.
        
        Args:
            trg (torch.Tensor): The target input sequence.
        
        Returns:
            trg_mask (torch.Tensor): The mask for the target input sequence.
        """
        if device is None:
            device = get_device()
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        ).to(device)
        
        return trg_mask

    def forward(self, src, trg):
        """
        Forward method for the Transformer class.
        
        Args:
            src (torch.Tensor): The source input sequence.
            trg (torch.Tensor): The target input sequence.
        
        Returns:
            out (torch.Tensor): The output tensor from the transformer.
        """
        src = src.to(self.device)
        trg = trg.to(self.device)
        src_mask = self.make_src_mask(src,self.src_pad_idx,self.device)
        trg_mask = self.make_trg_mask(trg,self.device)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out