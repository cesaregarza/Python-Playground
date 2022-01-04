from typing import Optional
from .player import Player
from .deck import generate_deck
from .cards import Card

class GameInstance:

    def __init__(self, players:int):
        #The minimum number of players is 2, the maximum number is the deck size divided by 6
        max_players = len(generate_deck()) // 6

        if (players < 2) or (players > max_players):
            raise ValueError(f"The number of players must be between 2 and {max_players}")

        self.players        = {i:Player() for i in range(players)}
        self.current_player = 0
        self.discard_pile   = []
        self.play_pile      = []
        self.deck           = generate_deck()
        self.normal_order   = True
    
    @property
    def top_card(self) -> Optional[Card]:
        """Get the top card, if there is one

        Returns:
            Optional[Card]: The top card, None if the pile is empty
        """
        if len(self.play_pile) == 0:
            return None
        return self.play_pile[-1]
    
    def draw_cards(self, draws:int = 1) -> list[Card]:
        """Draw cards from the deck, if there are any.
        Args:
            draws (int): The number of cards to draw
        
        Returns:
            list[Card]: The drawn cards
        """
        drawn_cards = self.deck[:draws]
        self.deck   = self.deck[draws:]
        return drawn_cards
    
    def deal_initial_cards(self) -> None:
        """Deal the initial 6 cards to each player"""
        for player_index, player in self.players.items():
            #Draw 6 cards
            drawn_cards = self.draw_cards(6)
            player.initial_draw(drawn_cards)
    
    def play_cards(self, cards:list[Card]) -> None:
        """Play one or more cards

        Args:
            cards (list[Card]): The cards to play
        """
        #Check if the cards are of the same rank
        if len(cards) > 1:
            if not all(card.rank == cards[0].rank for card in cards):
                raise ValueError("Cards must be of the same rank")
        
        first_card = cards[0]
        #Check if  card is playable
        if not self.card_playable(first_card):
            raise ValueError("The card is not playable")
        
        #If the rank played is stackable, multiply the action by the number of cards played
        actions = first_card.action()
        if first_card.stackable:
            for card in cards[1:]:
                actions += card.action()
        
        #Play the card and execute their action
        for action in actions:
            action.execute(self)
        
        #Assume the actions are valid and have finished, so we're done
        return None
    
    def card_playable(self, card:Card) -> bool:
        """Check if a card is playable

        Args:
            card (Card): The card to check
        
        Returns:
            bool: True if the card is playable, False otherwise
        """
        if self.top_card is None:
            return True
        
        #Check if the card is playable
        if self.normal_order:
            return self.top_card.rank <= card.rank
        else:
            return self.top_card.rank >= card.rank