from typing import Any
from .constants import ValueToCard, Suit
from .action_class import Action, ActionQueue
from .actions import *

class Card:
    def __init__(self, value:int, suit:Suit):
        self.value      = value
        self.suit       = suit
        self.stackable  = False
    
    def __repr__(self):
        card_name = ValueToCard.get_card_name(self.value)
        return f'{card_name} of {self.suit.name}'
    
    def __lt__(self, other: 'Card') -> bool:
        return self.value < other.value
    
    def __eq__(self, other: 'Card') -> bool:
        return self.value == other.value
    
    def __gt__(self, other: 'Card') -> bool:
        return self.value > other.value
    
    def __lte__(self, other: 'Card') -> bool:
        return self.value <= other.value
    
    def __gte__(self, other: 'Card') -> bool:
        return self.value >= other.value
    
    def __ne__(self, other: 'Card') -> bool:
        return self.value != other.value
    
    def __getattr__(self, name:str) -> int:
        try:
            getattr(self, name)
        except AttributeError:
            raise AttributeError(f'{name} is not a valid attribute of Card')
    
    def __setattr__(self, name:str, value:Any) -> None:
        try:
            setattr(self, name, value)
        except AttributeError:
            raise AttributeError(f'{name} is not a valid attribute of Card')
    
    def action(self) -> ActionQueue:
        return None
    
    @property
    def rank(self) -> int:
        return self.value

class Ace(Card):
    def __init__(self, suit:Suit):
        super().__init__(1, suit)
    
    def action(self) -> ActionQueue:
        return ActionQueue(AceAction())