'''The single-agent transformer'''

import logging
from enum import Enum, auto
from functools import partial
from multiprocessing import Process
from multiprocessing import SimpleQueue as Queue
from time import sleep

import numpy as np
from icecream import ic

from collision import test_collision
from messaging import MessageTypes, Msg, SenderTypes
from single_agent import Agent, Action


def test_conflict(action_1, action_2) -> bool:
  '''Test if two actions collide. The oracle function'''
  action_1, state_1 = action_1
  action_2, state_2 = action_2
  time_1 = action_1.time
  time_2 = action_2.time
  return test_collision((state_1, action_1.vel_vec, time_1), (state_2, action_2.vel_vec, time_2),
                        Agent.RADIUS)


class TransformerIState(Enum):
  '''An enum describing the transformer states'''
  AtGoal = auto()
  Conflicted = auto()
  RevertingFollow = auto()
  RevertingLeader = auto()
  WaitAcknowledge = auto()
  WaitAction = auto()


ZERO_ACTION = Action(-1, np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]), 0.0)


class TransformerState:
  '''The state of another transformer'''

  def __init__(self, pose, rev_path, curr_action, running, start_tick, queue, conflicted):
    self.rev_path = rev_path
    self.curr_action = curr_action
    self.running = running
    self.start_tick = start_tick
    self.queue = queue
    self.conflicted = conflicted
    self.at_goal = False
    self.pose = pose


class Transformer:
  '''The transformer'''
  TICK = 0.2
  WORLD_TICK = 0.1

  def __init__(self,
               initial_state: (float, float),
               goal_state: (float, float),
               agent_ctor,
               action_velocities=((-0.5, 0.5), (-0.5, 0.5)),
               action_time_max=0.5):
    self.transformers = {}
    self.state = TransformerIState.WaitAction
    self.reverse_actions = []
    self.queue = Queue()
    self.thread = None
    self.agent = agent_ctor(self, initial_state, goal_state, action_velocities, action_time_max)
    self.other_starts_goals = []
    self.log = logging.getLogger(f'tf-{self.agent.id}')
    self.need_acknowledgement = set()
    self.conflicts_with = set()
    self.current_action = None
    self.is_leader = self.get_id() == 0
    self.tick_count = 0
    self.reverse_schedule = None
    self.awaited_reversions = None
    self.pose = self.agent.pose

  def compute_reverse_schedule(self, action_log):
    '''Figure out how far back to go and what actions can run in parallel'''
    conflicted_agents = self.conflicts_with.copy()
    back_idx = -1
    while conflicted_agents:
      layer = action_log[back_idx]

    return []

  def update_ticks(self, _, count):
    '''Update the internal tick counter'''
    self.tick_count = max(count, self.tick_count)

  def register_transformer(self, transformer):
    '''Register a new transformer'''
    self.transformers[transformer.get_id()] = TransformerState(transformer.get_start(), [],
                                                               None, False, None,
                                                               transformer.get_queue(), False)
    self.other_starts_goals.append(transformer.get_start())
    self.other_starts_goals.append(transformer.get_goal())

  def get_start(self):
    '''Return the agent's start'''
    return self.agent.pose

  def get_goal(self):
    '''Return the agent's goal'''
    return self.agent.goal

  def get_id(self) -> int:
    '''Getter for this transformer's ID (same as its agent's ID)'''
    return self.agent.id

  def get_queue(self):
    '''Getter for this transformer's queue'''
    return self.queue

  def get_agent(self):
    '''Getter for this transformer's agent'''
    return self.agent

  def start(self):
    '''Start the process'''
    if not self.thread:
      other_starts_goals = np.vstack(self.other_starts_goals)
      self.thread = Process(target=self.run, args=(self.queue, other_starts_goals))
      self.thread.start()

    return self.thread

  def stop(self):
    '''Stop the process'''
    if self.thread and self.thread.is_alive():
      self.state = TransformerIState.AtGoal
      self.thread.join()

  def agent_request(self, other_starts_goals, _, action_req):
    '''Handle an action request from an agent'''
    if self.state is not TransformerIState.WaitAction:
      self.log.warning('Ignoring action request; mode is %s', self.state)
      return
    self.log.info('Received agent request')

    self.conflicts_with.clear()
    action, start_pose = action_req
    self.current_action = action_req
    for step in np.arange(Transformer.WORLD_TICK, action.time, Transformer.WORLD_TICK):
      pose = start_pose + step * action.vel_vec
      pose_array = np.full_like(other_starts_goals, pose)
      # Check that the action doesn't conflict with starts/goals
      if np.linalg.norm(pose_array - other_starts_goals,
                        -np.inf) <= Agent.GOAL_THRESHOLD + Agent.RADIUS:
        # We come too close to some start or goal; reject
        self.log.debug('Agent request conflicts with start or goal; rejecting')
        self.agent.queue.put(
            Msg(SenderTypes.Transformer, MessageTypes.ActionReject, self.get_id(), action_req))
        return

      # Check that the action doesn't conflict with known current/potential actions
      for (tf_id, transformer) in self.transformers.items():
        if test_conflict((action, pose), (ZERO_ACTION, transformer.pose)):
          # We'd run into where an agent currently is
          self.agent.queue.put(
              Msg(SenderTypes.Transformer, MessageTypes.ActionReject, self.get_id(), action_req))
          return

        if transformer.curr_action:
          tf_action, tf_pose = transformer.curr_action
          # if transformer.running:
          #   ticks_elapsed = self.tick_count - transformer.start_tick
          #   tf_pose += tf_action.vel_vec * ticks_elapsed * Transformer.WORLD_TICK

          if test_conflict((action, pose), (tf_action, tf_pose)):
            # The action conflicts with a currently executing or approved action
            self.conflicts_with.add(tf_id)
            self.log.debug('Agent request conflicts with approved action; rejecting')
            self.agent.queue.put(
                Msg(SenderTypes.Transformer, MessageTypes.ActionReject, self.get_id(), action_req))
            return

    # We can skip the reverse schedulability check because this is the special case where actions
    # are exactly reversible
    # Check with the other transformers
    self.need_acknowledgement = set(
        tf_id for tf_id in self.transformers if not self.transformers[tf_id].at_goal)
    if self.need_acknowledgement:
      self.log.debug('Sending out for approval')
      self.send_to_all(
          Msg(SenderTypes.Transformer, MessageTypes.ActionRequest, self.get_id(), action_req))
      self.log.debug('Need approval from %s', ic.format(self.need_acknowledgement))
      self.state = TransformerIState.WaitAcknowledge
    else:
      self.start_action()

  def send_to_all(self, msg):
    '''Send the same message to all transformers'''
    for transformer in self.transformers.values():
      if not transformer.at_goal:
        transformer.queue.put(msg)

  def agent_blocked(self, _id, _data):
    '''Handle the agent signalling they have no more actions to suggest'''
    self.state = TransformerIState.Conflicted
    self.send_to_all(Msg(SenderTypes.Transformer, MessageTypes.Conflicted, self.get_id()))

  def agent_action_done(self, _id, data):
    '''Handle the agent completing an action'''
    if self.state not in (TransformerIState.RevertingFollow, TransformerIState.RevertingLeader):
      self.state = TransformerIState.WaitAction
      self.current_action = None
      self.pose = data
    self.send_to_all(
        Msg(SenderTypes.Transformer, MessageTypes.ActionDone, self.get_id(), self.pose))

  def agent_at_goal(self, _id, pose):
    '''Handle the agent reaching its goal'''
    self.log.success('Reached goal!')
    self.state = TransformerIState.AtGoal
    self.pose = pose
    self.send_to_all(Msg(SenderTypes.Transformer, MessageTypes.AtGoal, self.get_id(), pose))

  def tf_request(self, tf_id, req):
    '''Handle a transformer requesting an action'''
    if self.state in (TransformerIState.RevertingFollow, TransformerIState.RevertingLeader):
      self.log.warning('Received action request from %d while reverting. Ignoring it...', tf_id)
      return

    self.log.debug('Received request from %d', tf_id)
    transformer = self.transformers[tf_id]
    # Check for conflict
    if self.current_action and test_conflict(self.current_action, req):
      # The actions are incompatible!
      self.log.notice('Action from %d conflicts with my action', tf_id)
      if self.get_id() > tf_id:
        self.log.notice('Approving action from %d and entering Conflicted', tf_id)
        transformer.queue.put(
            Msg(SenderTypes.Transformer, MessageTypes.ActionAccept, self.get_id()))
        self.transformers[tf_id].curr_action = req
        self.transformers[tf_id].running = False
        self.conflicts_with.add(tf_id)
        self.state = TransformerIState.Conflicted
        self.send_to_all(Msg(SenderTypes.Transformer, MessageTypes.Conflicted, self.get_id()))
      else:
        self.log.debug('Rejecting action from %d', tf_id)
        transformer.queue.put(
            Msg(SenderTypes.Transformer, MessageTypes.ActionReject, self.get_id()))

      return

    if test_conflict((ZERO_ACTION, self.pose), req):
      self.log.debug('Rejecting action from %d', tf_id)
      transformer.queue.put(Msg(SenderTypes.Transformer, MessageTypes.ActionReject, self.get_id()))
      return

    transformer.queue.put(Msg(SenderTypes.Transformer, MessageTypes.ActionAccept, self.get_id()))
    self.transformers[tf_id].curr_action = req
    self.transformers[tf_id].running = False

  def tf_reject(self, tf_id, _data):
    '''Handle an action being rejected'''
    self.log.debug('Action rejected by %d', tf_id)
    self.agent.queue.put(
        Msg(SenderTypes.Transformer, MessageTypes.ActionReject, self.get_id(), self.current_action))
    self.current_action = None
    self.state = TransformerIState.WaitAction

  def start_action(self):
    '''Utility method to trigger an approved action'''
    self.log.debug('Starting action')
    self.agent.queue.put(
        Msg(SenderTypes.Transformer, MessageTypes.ActionAccept, self.get_id(),
            self.current_action[0]))
    self.send_to_all(Msg(SenderTypes.Transformer, MessageTypes.ActionStart, self.get_id()))
    self.state = TransformerIState.WaitAction

  def tf_accept(self, tf_id, _):
    '''Handle an action being accepted'''
    self.log.debug('Action accepted by %d', tf_id)
    self.need_acknowledgement.discard(tf_id)
    if not self.need_acknowledgement and self.state is TransformerIState.WaitAcknowledge:
      self.reverse_actions.append((self.current_action, self.tick_count))
      self.start_action()

  def tf_cancel(self, tf_id, _):
    '''Handle another transformer cancelling an action request'''
    self.transformers[tf_id].curr_action = None
    self.transformers[tf_id].running = False
    if self.state is TransformerIState.Conflicted and tf_id in self.conflicts_with:
      self.conflicts_with.remove(tf_id)
      if not self.conflicts_with:
        self.state = TransformerIState.WaitAcknowledge
        self.start_action()

  def tf_at_goal(self, tf_id, pose):
    '''Handle another transformer reaching its goal'''
    self.is_leader = tf_id == self.get_id() - 1
    if tf_id in self.conflicts_with:
      self.conflicts_with.remove(tf_id)
      if not self.conflicts_with:
        self.start_action()

    self.transformers[tf_id].at_goal = True
    self.transformers[tf_id].pose = pose
    # A transformer reaching its goal may cause an action with pending accepts to be ready
    self.tf_accept(tf_id, None)

  def tf_action_start(self, tf_id, _):
    '''Handle another transformer starting an action'''
    self.transformers[tf_id].running = True
    self.transformers[tf_id].start_tick = self.tick_count
    self.transformers[tf_id].rev_path.append(
        (self.transformers[tf_id].curr_action, self.tick_count))

  def tf_action_done(self, tf_id, pose):
    '''Handle another transformer finishing an action'''
    self.transformers[tf_id].running = False
    self.transformers[tf_id].start_tick = None
    self.transformers[tf_id].curr_action = None
    self.transformers[tf_id].pose = pose
    if self.awaited_reversions:
      self.awaited_reversions.discard(tf_id)

  def tf_conflicted(self, tf_id, _):
    '''Handle another transformer announcing it is conflicted'''
    self.transformers[tf_id].conflicted = True
    # Note: This is only to make the formatting happier
    self_conflicted = self.state is TransformerIState.Conflicted
    if self_conflicted and all(transformer.conflicted for transformer in self.transformers):
      if self.is_leader:
        self.send_to_all(Msg(SenderTypes.Transformer, MessageTypes.StartReversion, self.get_id()))
        self.state = TransformerIState.RevertingLeader
        self.reverse_schedule = self.compute_reverse_schedule(
            [self.reverse_actions] +
            [self.transformers[tf_id].rev_path for tf_id in self.transformers])

  def tf_reversion_start(self, _id, _data):
    '''Handle the leader signalling the start of reversion'''
    self.state = TransformerIState.RevertingFollow

  def tf_reversion_end(self, _id, _data):
    '''Handle the leader signalling the end of reversion'''
    self.state = TransformerIState.WaitAction

  def tf_reversion_step(self, _id, reverse_action):
    '''Handle the leader requiring a reverse action step'''
    self.agent.queue.put(
        Msg(SenderTypes.Transformer, MessageTypes.StepReversion, self.get_id(), reverse_action))

  def run(self, queue, other_starts_goals):
    '''The main loop'''
    handlers = {
        (SenderTypes.Simulator, MessageTypes.TickUpdate):
            self.update_ticks,
        (SenderTypes.Agent, MessageTypes.ActionRequest):
            partial(self.agent_request, other_starts_goals),
        (SenderTypes.Agent, MessageTypes.NoValidActions):
            self.agent_blocked,
        (SenderTypes.Agent, MessageTypes.ActionDone):
            self.agent_action_done,
        (SenderTypes.Agent, MessageTypes.AtGoal):
            self.agent_at_goal,
        (SenderTypes.Transformer, MessageTypes.ActionRequest):
            self.tf_request,
        (SenderTypes.Transformer, MessageTypes.ActionReject):
            self.tf_reject,
        (SenderTypes.Transformer, MessageTypes.ActionAccept):
            self.tf_accept,
        (SenderTypes.Transformer, MessageTypes.ActionCancelled):
            self.tf_cancel,
        (SenderTypes.Transformer, MessageTypes.AtGoal):
            self.tf_at_goal,
        (SenderTypes.Transformer, MessageTypes.ActionStart):
            self.tf_action_start,
        (SenderTypes.Transformer, MessageTypes.ActionDone):
            self.tf_action_done,
        (SenderTypes.Transformer, MessageTypes.Conflicted):
            self.tf_conflicted,
        (SenderTypes.Transformer, MessageTypes.StartReversion):
            self.tf_reversion_start,
        (SenderTypes.Transformer, MessageTypes.EndReversion):
            self.tf_reversion_end,
        (SenderTypes.Transformer, MessageTypes.StepReversion):
            self.tf_reversion_step
    }

    while self.state is not TransformerIState.AtGoal:
      while not queue.empty():
        msg = queue.get()
        msg.dispatch(handlers)

      if self.state is TransformerIState.RevertingLeader:
        if not self.reverse_schedule:
          # TODO: Need to handle the case where the leader had to revert N steps
          self.start_action()
          self.send_to_all(Msg(SenderTypes.Transformer, MessageTypes.EndReversion, self.get_id()))
        elif not self.awaited_reversions:
          reversion_layer = self.reverse_schedule.pop()
          self.awaited_reversions = {action.agent_id for action in reversion_layer}
          for action in reversion_layer:
            self.transformers[action.agent_id].queue.put(
                Msg(SenderTypes.Transformer, MessageTypes.StepReversion, self.get_id(),
                    Action(action.agent_id, action.reverse, [], action.time)))

      sleep(Transformer.TICK)


class NullTransformer(Transformer):
  '''A fake transformer to make it easier to use non-transformed controllers for comparison'''

  def agent_request(self, _other_starts_goals, _, action_req):
    self.agent.queue.put(
        Msg(SenderTypes.Transformer, MessageTypes.ActionAccept, self.get_id(), action_req[0]))
