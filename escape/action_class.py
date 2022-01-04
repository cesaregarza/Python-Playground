from abc import ABC, abstractmethod
from typing import Union
from .game import GameInstance

class Action(ABC):

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def execute(self, game_instance: 'GameInstance') -> None:
        pass
    
    @abstractmethod
    def __repr__(self):
        pass

    def __add__(self, other:Union['Action', 'ActionQueue', list['Action']]) -> 'ActionQueue':
        if isinstance(other, list):
            return ActionQueue([self, *other])
        elif isinstance(other, ActionQueue):
            return other + self
        else:
            return ActionQueue([self, other])

class ActionQueue:
    def __init__(self, actions:list[Action]) -> None:
        self.actions = actions
    
    def __add__(self, other:Union[list[Action], Action, 'ActionQueue']) -> 'ActionQueue':
        if isinstance(other, list):
            return ActionQueue(self.actions + other)
        elif isinstance(other, ActionQueue):
            return ActionQueue(self.actions + other.actions)
        else:
            return ActionQueue(self.actions + [other])
    
    def __repr__(self) -> str:
        stri = "ActionQueue([\n"
        for action in self.actions:
            stri += f"\t{action.name},\n"
        stri += "])"
        return stri
    
    #Define an iterator through the actions using yield
    def __iter__(self) -> 'ActionQueue':
        return self
    
    def __next__(self) -> Action:
        if self.actions:
            yield self.actions.pop(0)
        else:
            raise StopIteration