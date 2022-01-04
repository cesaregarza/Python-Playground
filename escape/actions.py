from .action_class import Action, ActionQueue
from .constants import ValueToCard, Suit
from .game import GameInstance

class AceAction(Action):
    def execute(self):
        
        #Generate and return the action that corresponds to the ace
        def f(game_instance:GameInstance) -> GameInstance:
            pass
    