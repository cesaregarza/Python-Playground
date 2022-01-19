from abc import ABC, abstractmethod
import numpy as np

class Cypher(ABC):

    def __init__(self, cypher_map:dict[str, str], invert_map:bool = False) -> None:
        if invert_map:
            self.cypher_map = self.invert_map(cypher_map)
        else:
            self.cypher_map     = cypher_map
        self.plausibility   = None
    
    @staticmethod
    def invert_map(dictionary: dict) -> dict:
        return {v: k for k, v in dictionary.items()}

    @abstractmethod
    def encode(self, plaintext:str) -> str:
        pass

    @abstractmethod
    def decode(self, ciphertext:str) -> str:
        pass
    
    @abstractmethod
    def copy(self) -> 'Cypher':
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> 'Cypher':
        pass
    
    @staticmethod
    @abstractmethod
    def generate_cypher(character_list:list[str]) -> 'Cypher':
        pass


class SubstitutionCypher(Cypher):
    
    @staticmethod
    def generate_cypher(character_list:list[str]) -> 'SubstitutionCypher':
        """Generates a cypher using the given character list

        Args:
            character_list (list[str]): allowable character list

        Returns:
            Cypher: generated cypher
        """
        cypher_map = {}
        character_list_copy = character_list.copy()
        for char in character_list:
            cypher_map[char] = np.random.choice(character_list_copy)
            character_list_copy.remove(cypher_map[char])

        return SubstitutionCypher(cypher_map)
    
    def decode(self, text:str) -> str:
        """Encodes the given text using the cypher map

        Args:
            text (str): text to encode

        Returns:
            str: encoded text
        """
        encoded_text = ""
        for char in text:
            encoded_text += self.cypher_map[char]
        return encoded_text
    
    def encode(self, text:str) -> str:
        """Decodes the given text using the cypher map

        Args:
            text (str): text to decode

        Returns:
            str: decoded text
        """
        decoded_text = ""
        char_map = self.invert_map(self.cypher_map)
        for char in text:
            decoded_text += char_map[char]
        return decoded_text
    
    def __getitem__(self, char:str) -> str:
        return self.cypher_map[char]
    
    def __setitem__(self, char:str, value:str) -> None:
        self.cypher_map[char] = value
    
    def swap_chars(self, char1:str, char2:str) -> None:
        """Swaps the characters in the cypher map

        Args:
            char1 (str): first character
            char2 (str): second character
        """
        self.cypher_map[char1], self.cypher_map[char2] = self.cypher_map[char2], self.cypher_map[char1]
    
    def __eq__(self, other:'SubstitutionCypher') -> bool:
        if not isinstance(other, SubstitutionCypher):
            return False
        return self.cypher_map == other.cypher_map
    
    def copy(self) -> 'SubstitutionCypher':
        """Returns a copy of the cypher

        Returns:
            Cypher: copy of the cypher
        """
        new_cypher              = SubstitutionCypher(self.cypher_map.copy())
        new_cypher.plausibility = self.plausibility
        return new_cypher
    
    def keys(self) -> list[str]:
        """Returns the keys of the cypher map

        Returns:
            list[str]: keys of the cypher map
        """
        return list(self.cypher_map.keys())
    
    def update(self, base_cypher: 'SubstitutionCypher', *args, **kwargs) -> 'SubstitutionCypher':
        """Updates the cypher with the given base cypher

        Args:
            base_cypher (Cypher): base cypher

        Returns:
            Cypher: updated cypher
        """
        copy    = self.copy()
        chars   = base_cypher.keys()
        char1   = np.random.choice(chars)
        char2   = np.random.choice([char for char in chars if char != char1])
        copy.swap_chars(char1, char2)
        return copy

class EnigmaCypher(Cypher):
    
    @staticmethod
    def generate_cypher(character_list:list[str]) -> 'EnigmaCypher':
        pass