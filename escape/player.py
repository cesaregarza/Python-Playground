from .cards import Card

class Player:
    def __init__(self) -> None:
        self.hand               = []
        self.playable_cards     = []
    
    def initial_draw(self, cards:list[Card]) -> None:
        self.hand = cards