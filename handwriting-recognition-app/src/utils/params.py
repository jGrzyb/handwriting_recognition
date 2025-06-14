import time


class Params:
    vocab = "abcdefghijklmnopqrstuvwxyzz"  # Example vocabulary
    sos_token = 27  # Start of sequence token
    eos_token = 28  # End of sequence token

    @staticmethod
    def encode_string(s):
        return [Params.vocab.index(char) for char in s if char in Params.vocab]

    @staticmethod
    def decode_string(indices):
        return ''.join([Params.vocab[i] for i in indices])


