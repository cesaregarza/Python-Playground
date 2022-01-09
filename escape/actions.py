from .action_class import Action, ActionQueue
from .constants import ValueToCard, Suit
from .game import GameInstance

class AceAction(Action):
    def execute(self, game_instance:GameInstance, skips:int) -> GameInstance:
        game_instance.current_player = (game_instance.current_player + skips) % game_instance.num_players
        return game_instance

class SkipPlayer(Action):
    def execute(self, game_instance:GameInstance) -> GameInstance:
        game_instance.current_player = (game_instance.current_player + 1) % game_instance.num_players
        return game_instance

class NextCardLower(Action):
    def execute(self, game_instance:GameInstance) -> GameInstance:
        game_instance.normal_order = False
        return game_instance

class BanishPlayPile(Action):
    def execute(self, game_instance:GameInstance) -> GameInstance:
        game_instance.play_pile = []
        return game_instance