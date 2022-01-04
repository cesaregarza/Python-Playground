from enum import Enum, auto

class ValueToCard:

    conversion_table = {
        0: 'King',
        1: 'Ace',
        2: '2',
        3: '3',
        4: '4',
        5: '5',
        6: '6',
        7: '7',
        8: '8',
        9: '9',
        10: '10',
        11: 'Jack',
        12: 'Queen',
    }

    inverse_conversion_table    = {v:k for k,v in conversion_table.items()}
    max_value                   = max(conversion_table.keys())
    
    @staticmethod
    def get_card_name(value:int) -> str:
        new_value = value % (ValueToCard.max_value + 1)
        return ValueToCard.conversion_table[new_value]
    
    @staticmethod
    def get_card_value(card_name:str) -> int:
        return ValueToCard.inverse_conversion_table[card_name]

class Suit(Enum):
    Clubs       = auto()
    Diamonds    = auto()
    Hearts      = auto()
    Spades      = auto()