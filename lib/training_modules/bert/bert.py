from abc import ABC


class Bert(ABC):
    def get_tokens_from_masked_text(self, masked_text):
        pass

    def get_ids_from_tokens(self, tokens):
        pass
