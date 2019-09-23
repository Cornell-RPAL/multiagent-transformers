'''Single-agent naive controllers'''

import logging
from collections import namedtuple
from enum import Enum, auto
from functools import partial
from multiprocessing import Process
from multiprocessing import SimpleQueue as Queue
from random import shuffle
from time import sleep

import numpy as np
from icecream import ic

from messaging import MessageTypes, Msg, SenderTypes

Action = namedtuple('Action', ['agent_id', 'vel_vec', 'reverse', 'time'])


class AgentState(Enum):
  '''The single-agent controller states'''
  Waiting = auto()
  ActionRequested = auto()
  ExecutingAction = auto()
  GoalReached = auto()
  RevertStep = auto()
  RevertWaiting = auto()


class Agent:
  '''Abstract parent class for a single-agent controller'''
  id_counter = 0
  GOAL_THRESHOLD = 0.05
  RADIUS = 0.1
  TICK = 0.2

  def __init__(self,
               transformer,
               initial_state: (float, float),
               goal_state: (float, float),
               action_velocities=((-0.5, 0.5), (-0.5, 0.5)),
               action_time_max=0.5):
    self.transformer = transformer
    self.id = Agent.id_counter
    Agent.id_counter += 1
    self.world = None
    self.goal = np.array(goal_state)
    self.state = AgentState.Waiting
    self.reverse_actions = []
    self.pose = np.array(initial_state)
    self.invalid_actions = set()
    x_velocities = np.linspace(*action_velocities[0])
    y_velocities = np.linspace(*action_velocities[1])
    action_times = np.linspace(0.0, action_time_max)
    self.action_velocities = {(x, y, t) for x in x_velocities for y in y_velocities
                              for t in action_times}
    self.queue = Queue()
    self.thread = None
    self.log = logging.getLogger(f'agent-{self.id}')

  def start(self):
    '''Start running'''
    if not self.thread:
      self.thread = Process(
          target=self.run, args=(self.queue, self.world.action_queue, self.transformer.queue))
      self.thread.start()

    return self.thread

  def stop(self):
    '''Stop running'''
    if self.thread and self.thread.is_alive():
      self.state = AgentState.GoalReached
      self.thread.join()

  def get_queue(self):
    '''Getter for the agent's message queue'''
    return self.queue

  def at_goal(self) -> bool:
    '''Check if the agent is at its goal'''
    self.log.info('Distance to goal: %f', np.linalg.norm(self.pose - self.goal))
    return np.linalg.norm(self.pose - self.goal) <= Agent.GOAL_THRESHOLD

  def set_world(self, world):
    '''Setter for self.world'''
    self.world = world

  def generate_action(self):
    '''Abstract method to create a potentially valid action'''
    return Action(self.id, np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0)

  def propose_action(self, tf_queue):
    '''Request an action from the transformer'''
    action_candidate = self.generate_action()
    if action_candidate:
      tf_queue.put(
          Msg(SenderTypes.Agent, MessageTypes.ActionRequest, self.id,
              (action_candidate, self.pose)))
      self.state = AgentState.ActionRequested
    else:
      # No actions remain
      tf_queue.put(Msg(SenderTypes.Agent, MessageTypes.NoValidActions, self.id))

  def run_action(self, action, action_queue):
    '''Start an action'''
    action_queue.put(action)
    self.state = AgentState.ExecutingAction

  def reject_action(self, _, action):
    '''Receive an action rejection'''
    self.log.debug('Action rejected')
    action = action[0]
    self.invalid_actions.add((action.vel_vec[0, 0], action.vel_vec[0, 1], action.time))
    self.state = AgentState.Waiting

  def accept_action(self, action_queue, _, action):
    '''Receive an action acceptance'''
    self.log.debug('Action approved')
    self.reverse_actions.append(Action(self.id, action.reverse, [], action.time))
    self.run_action(action, action_queue)
    self.invalid_actions.clear()

  def action_done(self, tf_queue, _, new_pose):
    '''Receive an ActionDone message from the simulator'''
    self.pose = new_pose
    tf_queue.put(Msg(SenderTypes.Agent, MessageTypes.ActionDone, self.id, self.pose))
    if self.state is not AgentState.RevertWaiting:
      self.state = AgentState.Waiting

  def interrupted(self, _id, _data):
    '''Receive an interruption'''
    self.state = AgentState.RevertWaiting
    self.log.notice('Interrupted')

  def end_revert(self, _id, _data):
    '''Stop reverting'''
    self.state = AgentState.Waiting
    self.log.notice('Done reverting')

  def step_revert(self, _id, _data):
    '''Signal a reversion step'''
    self.state = AgentState.RevertStep

  def run(self, queue, action_queue, tf_queue):
    '''Main controller loop'''
    handlers = {
        (SenderTypes.Simulator, MessageTypes.ActionDone):
            partial(self.action_done, tf_queue),
        (SenderTypes.Transformer, MessageTypes.ActionReject):
            self.reject_action,
        (SenderTypes.Transformer, MessageTypes.ActionAccept):
            partial(self.accept_action, action_queue),
        (SenderTypes.Transformer, MessageTypes.Interruption):
            self.interrupted,
        (SenderTypes.Transformer, MessageTypes.EndReversion):
            self.end_revert,
        (SenderTypes.Transformer, MessageTypes.StepReversion):
            self.step_revert
    }

    while self.state is not AgentState.GoalReached:
      # Get messages if there are any
      while not queue.empty():
        msg = queue.get()
        msg.dispatch(handlers)

      if self.state is AgentState.RevertStep:
        self.run_action(self.reverse_actions.pop(), action_queue)
        self.log.debug('Ran reverse step')
        self.state = AgentState.RevertWaiting
      elif self.at_goal():
        self.state = AgentState.GoalReached
        tf_queue.put(Msg(SenderTypes.Agent, MessageTypes.AtGoal, self.id, self.pose))
        self.log.success('Reached goal!')
      elif self.state is AgentState.Waiting:
        self.propose_action(tf_queue)
        self.log.debug('Requested action')

      sleep(Agent.TICK)


class RandomProgressAgent(Agent):
  '''An Agent type that picks a random action making progress toward the goal'''

  def generate_action(self):
    # Pick a random vector making non-zero progress toward the goal
    goal_vec = self.goal - self.pose
    action_candidates = self.action_velocities - self.invalid_actions
    shuffle(list(action_candidates))
    for action in action_candidates:
      x, y, duration = action
      vel = np.array([[x, y]])
      if np.dot(vel, goal_vec.T) > 0:
        return Action(self.id, vel, np.array([-vel[0, 1], vel[0, 0]]), duration)

      self.invalid_actions.add(action)

    self.log.warning('No actions left!')
    return None


class MaximalProgressAgent(Agent):
  '''An Agent type that picks the best action for itself at every choice'''

  def generate_action(self):
    # Find the best valid action vector
    action_candidates = self.action_velocities - self.invalid_actions
    self.log.debug('%d possible candidates', len(action_candidates))
    if not action_candidates:
      self.log.warning('No actions left!')
      return None

    best_dist = np.inf
    best = None
    for action in action_candidates:
      x, y, action_time = action
      vel = np.array([[x, y]])
      end_pose = self.pose + vel * action_time
      end_dist = np.linalg.norm(self.goal - end_pose)
      if end_dist < best_dist:
        best_dist = end_dist
        best = action

    return Action(self.id, np.array([[best[0], best[1]]]), np.array([-best[1], best[0]]), best[2])
