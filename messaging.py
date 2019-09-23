'''Message definitions'''
from enum import Enum, auto
import logging


class SenderTypes(Enum):
  '''The types of senders'''
  Transformer = auto()
  Agent = auto()
  Simulator = auto()


class MessageTypes(Enum):
  '''The types of messages'''
  ActionAccept = auto()
  ActionAcknowledge = auto()
  ActionCancelled = auto()
  ActionDone = auto()
  ActionReject = auto()
  ActionRequest = auto()
  ActionStart = auto()
  NoValidActions = auto()

  AtGoal = auto()

  Conflicted = auto()

  Interruption = auto()
  StartReversion = auto()
  StepReversion = auto()
  EndReversion = auto()

  TickUpdate = auto()


class Msg:
  '''A message'''

  def __init__(self,
               sender_type: SenderTypes,
               message_type: MessageTypes,
               sender_id: int,
               data=None):
    self.sender_type = sender_type
    self.message_type = message_type
    self.sender_id = sender_id
    self.data = data
    self.log = logging.getLogger('messaging')

  def __str__(self):
    return f'({self.sender_type}, {self.message_type}): {self.sender_id} -> {self.data}'

  def dispatch(self, handlers):
    '''Takes a dictionary of patterns and handler functions and runs the right function for this
    message'''
    for pattern in handlers:
      sender_t, message_t = pattern
      if self.sender_type is sender_t and self.message_type is message_t:
        handlers[pattern](self.sender_id, self.data)
        return

    self.log.warning('Unhandled message! %s', self)
