import unittest
from fundaml.tokenizers import HFTokenizer

class TestHFTokenizers(unittest.TestCase):
    
    def test_encode_sentence_no_special_tokens(self):
        input_sequence = "Hello my name is Tony."
        tokenizer = HFTokenizer()
        print(tokenizer.get_vocab_size())
        tokenized = tokenizer.encode(input_sequence,add_special_tokens=False)
        self.assertEqual(
            [8667, 1139, 1271, 1110, 3270, 119],
            tokenized,
        )
        decoded = tokenizer.decode(tokenized, skip_special_tokens=True)
        print(decoded)  
        
    def test_encode_sentence_with_special_tokens(self):
        input_sequence = "Hello my name is Tony."
        tokenizer = HFTokenizer()
        tokenized = tokenizer.encode(input_sequence,add_special_tokens=True)
        self.assertEqual(
            [101, 8667, 1139, 1271, 1110, 3270, 119, 102],
            tokenized,
        )
        decoded = tokenizer.decode(tokenized, skip_special_tokens=False)
        print(decoded)          

    def test_encode_sentences(self):
        input_sequence = ["Hello my name is Tony.","I live in Canada."] 
        tokenizer = HFTokenizer()
        tokenized = tokenizer.encode(input_sequence,add_special_tokens=False)
        self.assertEqual(
            [[8667, 1139, 1271, 1110, 3270, 119],[146, 1686, 1107, 1803, 119, 0]],
            tokenized,
        )
        print([ tokenizer.decode(encoded) for encoded in tokenized])
        
        
if __name__ == "__main__":
    unittest.main()  