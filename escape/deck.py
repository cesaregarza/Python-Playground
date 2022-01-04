import numpy as np
from .cards import Card
from .constants import Suit

#Generate a deck of 52 cards
def generate_deck():
    suits = [Suit.Clubs, Suit.Diamonds, Suit.Hearts, Suit.Spades]
    ranks = [i for i in range(13)]
    deck = []
    for suit in suits:
        for rank in ranks:
            deck += [Card(rank, suit)]
    
    deck = np.array(deck)
    np.random.shuffle(deck)
    return deck