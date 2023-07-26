from transformers import AutoTokenizer

class HFTokenizer:
    """
    A class to wrap the Hugging Face tokenizer for easy use.

    Attributes:
    -----------
    checkpoint : str
        The name of the Hugging Face checkpoint.
    tokenizer : transformers.AutoTokenizer
        The Hugging Face tokenizer.

    Methods:
    --------
    encode(sentences, padding=True, truncation=True, max_length=512, add_special_tokens=True)
        Encodes a given list of sentences.
    get_vocab_size()
        Returns the vocabulary size of the tokenizer.
    decode(encoded, skip_special_tokens=True)
        Decodes a given encoded input.
    """
    def __init__(self, hf_checkpoint_name="distilbert-base-cased"):
        """
        Constructs all the necessary attributes for the TokenizerWrapper object.

        Parameters:
        -----------
        hf_checkpoint_name : str
            The name of the Hugging Face checkpoint. Default is "distilbert-base-cased".
        """
        self.checkpoint = hf_checkpoint_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

    def encode(self, sentences, padding=True, truncation=True, max_length=512, add_special_tokens=True):
        """
        Encodes a given list of sentences.

        Parameters:
        -----------
        sentences : str or List[str]
            The sentences to be encoded.
        padding : bool, optional
            Whether to pad the sentences. Default is True.
        truncation : bool, optional
            Whether to truncate the sentences. Default is True.
        max_length : int, optional
            The maximum length for the sentences. Default is 512.
        add_special_tokens : bool, optional
            Whether to add special tokens. Default is True.

        Returns:
        --------
        List[int] or List[List[int]]
            The encoded sentences. Returns a list of integers if only one sentence is given.
        """
        encoded = self.tokenizer(
            sentences, 
            padding=padding, 
            truncation=truncation, 
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            return_tensors="pt"
        ).input_ids.tolist()  

        return encoded[0] if len(encoded) == 1 else encoded

    def get_vocab_size(self):
        """
        Returns the vocabulary size of the tokenizer.

        Returns:
        --------
        int
            The vocabulary size.
        """
        return self.tokenizer.vocab_size

    def decode(self, encoded, skip_special_tokens=True):
        """
        Decodes a given encoded input.

        Parameters:
        -----------
        encoded : List[int]
            The encoded input to be decoded.
        skip_special_tokens : bool, optional
            Whether to skip special tokens during decoding. Default is True.

        Returns:
        --------
        str
            The decoded input.
        """
        return self.tokenizer.decode(encoded, skip_special_tokens=skip_special_tokens)
    
    
    
