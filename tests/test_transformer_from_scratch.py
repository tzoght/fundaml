from fundaml.transformer_from_scratch import Transformer, MultiHeadSelfAttentionBlock, TransformerBlock, Encoder, DecoderBlock, Decoder, Transformer, get_device
import unittest
import torch
import torch.nn as nn
from torch.autograd import Variable

class TestSelfAttention(unittest.TestCase):

    def test_init(self):
        # Test that the object initializes correctly
        try:
            self_attention = MultiHeadSelfAttentionBlock(embedding_size=512, num_heads=8)
        except Exception as e:
            self.fail(f"Initialization of SelfAttention failed with {e}")

    def test_forward_pass_small(self):
        # Create a SelfAttention object
        self_attention = MultiHeadSelfAttentionBlock(embedding_size=6, num_heads=2)
        self_attention.to(get_device())

        # Create dummy inputs for the forward pass
        # 64 examples, 20 tokens, 512 embedding size
        N, sequence_length = 1, 5  # Number of examples and sequence length
        values = torch.rand((N, sequence_length, 6))
        keys = torch.rand((N, sequence_length, 6))
        query = torch.rand((N, sequence_length, 6))
        mask = torch.ones((N, 1,1,sequence_length))

        # Test the forward pass
        try:
            output = self_attention(values, keys, query, mask)
        except Exception as e:
            self.fail(f"Forward pass of SelfAttention failed with {e}")


    def test_forward_pass(self):
        # Create a SelfAttention object
        self_attention = MultiHeadSelfAttentionBlock(embedding_size=512, num_heads=8)
        self_attention.to(get_device())

        # Create dummy inputs for the forward pass
        # 64 examples, 20 tokens, 512 embedding size
        N, sequence_length = 64, 20  # Number of examples and sequence length
        values = torch.rand((N, sequence_length, 512))
        keys = torch.rand((N, sequence_length, 512))
        query = torch.rand((N, sequence_length, 512))
        mask = torch.ones((N, 1,1,sequence_length))

        # Test the forward pass
        try:
            output = self_attention(values, keys, query, mask)
        except Exception as e:
            self.fail(f"Forward pass of SelfAttention failed with {e}")

    def test_output_shape(self):
        # Create a SelfAttention object
        self_attention = MultiHeadSelfAttentionBlock(embedding_size=512, num_heads=8)
        self_attention.to(get_device())

        # Create dummy inputs for the forward pass
        N, sequence_length = 64, 20  # Number of examples and sequence length
        values = torch.rand((N, sequence_length, 512))
        keys = torch.rand((N, sequence_length, 512))
        query = torch.rand((N, sequence_length, 512))
        mask = torch.ones((N, 1,1,sequence_length))

        # Perform the forward pass and check the output shape
        output = self_attention(values, keys, query, mask)
        self.assertEqual(output.shape, (N, sequence_length, 512),
                         "Output shape of SelfAttention forward pass is incorrect")


class TestTransformerBlock(unittest.TestCase):
    def setUp(self):
        self.embed_size = 512
        self.num_heads = 8
        self.dropout_rate = 0.1
        self.forward_expansion = 4
        self.transformer_block = TransformerBlock(
            self.embed_size, self.num_heads, self.dropout_rate, self.forward_expansion
        )
        self.transformer_block.to(get_device())

    def test_forward_pass(self):
        N = 64  # batch size
        seq_len = 10  # sequence length

        values = Variable(torch.rand(N, seq_len, self.embed_size))
        keys = Variable(torch.rand(N, seq_len, self.embed_size))
        queries = Variable(torch.rand(N, seq_len, self.embed_size))
        mask = Variable(torch.ones(N, 1, 1, seq_len))

        out = self.transformer_block(values, keys, queries, mask)
        
        # Assert that the output has the correct type and shape
        self.assertIsInstance(out, torch.Tensor)
        self.assertTupleEqual(out.shape, (N, seq_len, self.embed_size))


class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.src_vocab_size = 1000
        self.embed_size = 512
        self.num_layers = 6
        self.num_heads = 8
        self.forward_expansion = 4
        self.dropout_rate = 0.1
        self.max_length = 5000

        self.encoder = Encoder(
            src_vocab_size=self.src_vocab_size,
            embed_size=self.embed_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            forward_expansion=self.forward_expansion,
            dropout_rate=self.dropout_rate,
            max_length=self.max_length
        )

        self.encoder.to(get_device())

    def test_forward(self):
        batch_size = 32
        seq_length = 10
        src_vocab = torch.randint(0, self.src_vocab_size, (batch_size, seq_length)).to(get_device())
        src_mask = torch.zeros((batch_size, 1, 1, seq_length)).to(get_device())
        
        out = self.encoder(src_vocab, src_mask)

        self.assertTupleEqual(out.shape, (batch_size, seq_length, self.embed_size))


class TestDecoderBlock(unittest.TestCase):

    def setUp(self):
        self.batch_size = 64
        self.seq_length = 50
        self.embed_size = 512
        self.num_heads = 8
        self.forward_expansion = 4
        self.dropout_rate = 0.1

        # Initialize an instance of the DecoderBlock
        self.decoder_block = DecoderBlock(self.embed_size, self.num_heads, self.forward_expansion, self.dropout_rate)
        self.decoder_block.to(get_device())

        # Initialize some random test data
        self.x = torch.randn(self.batch_size, self.seq_length, self.embed_size)
        self.value = torch.randn(self.batch_size, self.seq_length, self.embed_size)
        self.key = torch.randn(self.batch_size, self.seq_length, self.embed_size)
        self.src_mask = torch.randn(self.batch_size, 1, 1, self.seq_length)
        self.trg_mask = torch.randn(self.batch_size, 1, 1, self.seq_length)

    def test_decoder_block(self):
        # Forward pass through the DecoderBlock
        output = self.decoder_block(self.x, self.value, self.key, self.src_mask, self.trg_mask)

        # Assert the output shape is as expected
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.embed_size))


class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.batch_size = 64
        self.seq_length = 50
        self.embed_size = 512
        self.num_heads = 8
        self.num_layers = 8
        self.forward_expansion = 4
        self.dropout_rate = 0.1
        self.trg_vocab_size = 10000
        self.max_length = 1000

        self.decoder = Decoder(
            self.trg_vocab_size,
            self.embed_size,
            self.num_layers,
            self.num_heads,
            self.forward_expansion,
            self.dropout_rate,
            self.max_length
        )
        self.decoder.to(get_device())

        self.x = torch.randint(0, self.trg_vocab_size, (self.batch_size, self.seq_length))
        self.enc_out = torch.randn(self.batch_size, self.seq_length, self.embed_size)
        self.src_mask = Transformer.make_src_mask(self.x, 0)
        torch.ones((self.batch_size, 1, 1, self.seq_length))
        self.trg_mask = Transformer.make_trg_mask(self.x)

    def test_forward(self):
        # Perform a forward pass through the decoder
        output = self.decoder(self.x, self.enc_out, self.src_mask, self.trg_mask)

        # Check the output size
        self.assertTupleEqual(output.size(), (self.batch_size, self.seq_length, self.trg_vocab_size))

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.src_pad_idx = 0
        self.trg_pad_idx = 0
        self.src_vocab_size = 10
        self.trg_vocab_size = 10
        self.model = Transformer(self.src_vocab_size, self.trg_vocab_size, self.src_pad_idx, self.trg_pad_idx)
        self.model.to(get_device())

        self.x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(get_device())
        self.trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(get_device())

    def test_forward_pass(self):
        out = self.model(self.x, self.trg[:, :-1])
        self.assertTupleEqual(out.shape, (self.trg.shape[0], self.trg.shape[1] - 1, self.trg_vocab_size))

if __name__ == "__main__":
    unittest.main()  