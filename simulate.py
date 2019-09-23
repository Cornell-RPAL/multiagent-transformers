'''The main simulation driver'''

import datetime
import logging
import sys
from multiprocessing import SimpleQueue
from random import choice
from threading import Thread
from time import sleep

import coloredlogs
import verboselogs
import fire
import numpy as np
from icecream import ic
import json

from collision import test_collision
from messaging import MessageTypes, Msg, SenderTypes
from single_agent import (Action, Agent, MaximalProgressAgent, RandomProgressAgent)
from transformer import NullTransformer, Transformer

LOG_FMT = '%(asctime)s %(name)s %(levelname)s: %(message)s'
LOG_FIELD_STYLES = {
    'asctime': {
        'color': 'green'
    },
    'hostname': {
        'color': 'magenta'
    },
    'levelname': {
        'color': 'white',
        'bold': True,
    },
    'name': {
        'color': 'blue'
    },
    'programname': {
        'color': 'cyan'
    }
}
LOG_BASE_LEVEL = 'INFO'
LOG_VERBOSE_LEVEL = 'DEBUG'

verboselogs.install()
coloredlogs.install(fmt=LOG_FMT, level='INFO', field_styles=LOG_FIELD_STYLES)


# TODO: This is a bit redundant; should be improved upon
class AgentSimState:
  '''Track agent simulation state'''

  def __init__(self, pose, goal, vel, action_time, queue):
    self.pose = pose
    self.goal = goal
    self.vel = vel
    self.action_time = action_time
    self.queue = queue

  def at_goal(self) -> bool:
    '''Check if the agent is at its goal'''
    return np.linalg.norm(self.pose - self.goal) <= Agent.GOAL_THRESHOLD


class World:
  '''A simulated world'''
  TICK = 0.1

  def __init__(self, agents, transformers):
    self.agents = agents
    self.transformers = transformers
    self.tf_queues = [[tf.get_queue(), True] for tf in transformers.values()]

    # Initialize each agent to be at its initial state and stationary
    self.agent_states = {
        agent_id: AgentSimState(agent.pose, agent.goal, np.zeros(2), 0.0, agent.get_queue())
        for (agent_id, agent) in self.agents.items()
    }

    self.moving_agents = set()
    self.state_logs = []
    self.log = logging.getLogger('world')
    self.action_queue = SimpleQueue()

  def simulate_step(self, state, vec, time, step: float):
    '''Simulate a single step forward'''
    state += vec * step
    time -= step

    return state, time

  def start_action(self, action: Action):
    '''Register an approved action'''
    agent_id = action.agent_id
    if agent_id not in self.moving_agents:
      self.log.debug('Starting action for agent %d', agent_id)
      self.agent_states[agent_id].vel = action.vel_vec
      self.agent_states[agent_id].action_time = action.time
      self.moving_agents.add(agent_id)
    else:
      self.log.warning("Tried to add action for agent %d; they're already moving!", agent_id)

  def monitor_actions(self):
    '''Thread worker to get actions from the queue'''
    for action in iter(self.action_queue.get, 'DONE'):
      self.start_action(action)

  def check_done(self):
    '''Check if there are any agents which haven't reached their goals'''
    all_done = True
    for (agent_id, agent) in self.agent_states.items():
      agent_done = agent.at_goal()
      self.tf_queues[agent_id][1] = not agent_done
      all_done = all_done and agent_done

    return all_done

  def output_logs(self, output_file: str):
    '''Write logs to a file'''
    self.log.info('Writing logs to %s', output_file)
    with open(output_file, 'w', newline='') as output:
      json.dump(self.state_logs, output)

  def simulate(self):
    '''Main simulation loop'''
    self.log.info('Starting main simulation loop')
    queue_thread = Thread(target=self.monitor_actions)
    queue_thread.start()
    agent_procs = [(transformer.start(), transformer.agent.start())
                   for transformer in self.transformers.values()]
    ticks = 0

    def wrap_up():
      self.log.info('Simulation done!')
      for (tf_proc, agent_proc) in agent_procs:
        tf_proc.terminate()
        agent_proc.terminate()

      self.action_queue.put('DONE')
      queue_thread.join()
      self.log.info('Agents shut down')

    while not self.check_done():
      # Test for collisions in the next tick and update movement
      done_moving_agents = set()
      # To avoid size changes from concurrent modification to the set
      currently_moving_agents = self.moving_agents.copy()
      for agent_id in currently_moving_agents:
        agent_state = self.agent_states[agent_id]
        self.log.verbose('Moving agent %d, %fs remaining', agent_id, agent_state.action_time)
        for other_id in self.agent_states:
          if agent_id == other_id:
            continue
          other_state = self.agent_states[other_id]
          if test_collision((agent_state.pose, agent_state.vel, agent_state.action_time),
                            (other_state.pose, other_state.vel, other_state.action_time),
                            Agent.RADIUS):
            self.log.error('Collision between agents %d and %d!', agent_id, other_id)
            wrap_up()
            return

        agent_pos, t_remaining = self.simulate_step(agent_state.pose, agent_state.vel,
                                                    agent_state.action_time,
                                                    min(World.TICK, agent_state.action_time))
        self.agent_states[agent_id].pose = agent_pos
        self.agent_states[agent_id].action_time = t_remaining
        if t_remaining <= 0.0:
          # Remove stopped agents and signal end of actions
          self.log.verbose('Done moving %d', agent_id)
          done_moving_agents.add(agent_id)
          self.agent_states[agent_id].action_time = 0.0
          self.agent_states[agent_id].vel = np.zeros(2)
          done_message = Msg(SenderTypes.Simulator, MessageTypes.ActionDone, -1, agent_pos)
          self.agent_states[agent_id].queue.put(done_message)

      self.moving_agents -= done_moving_agents

      # Log states
      self.state_logs.append({
          agent_id: {
              'pose': state.pose.tolist(),
              'vel': state.vel.tolist()
          } for (agent_id, state) in self.agent_states.items()
      })

      # Update tick count
      ticks += 1
      for (tf_q, running) in self.tf_queues:
        if running:
          tf_q.put(Msg(SenderTypes.Simulator, MessageTypes.TickUpdate, -1, ticks))

      sleep(World.TICK)

    wrap_up()


def load_scenario(scenario_file_path: str, use_transformers: bool):
  '''Load a simulation scenario (list of agents and start/goal positions) from a file'''
  with open(scenario_file_path, 'r') as scenario_file:
    scenario_data = json.load(scenario_file)

  if use_transformers:
    ctor = Transformer
  else:
    ctor = NullTransformer

  opt_args = {}
  if 'action_velocities' in scenario_data:
    act_vels = scenario_data['action_velocities']
    opt_args['action_velocities'] = ((act_vels['x']['low'], act_vels['x']['high']),
                                     (act_vels['y']['low'], act_vels['y']['high']))

  if 'action_time_max' in scenario_data:
    opt_args['action_time_max'] = scenario_data['action_time_max']

  agents = {}

  agent_ctors = [MaximalProgressAgent, RandomProgressAgent]
  for agent_spec in scenario_data['agents'].values():
    agent = ctor(agent_spec['start'], agent_spec['goal'], choice(agent_ctors), **opt_args)
    agents[agent.get_id()] = agent

  return agents


def generate_agents(num_agents: int, use_transformers: bool, scenario_file_path: str):
  '''Generate num_agents random agents'''
  if use_transformers:
    ctor = Transformer
  else:
    ctor = NullTransformer

  agent_ctors = [MaximalProgressAgent]
  agents = {}
  starts = []
  goals = []
  # We'll make an arena of 4 * num_agents * agent size along each dimension
  bound = 4 * num_agents * 2 * Agent.RADIUS
  # Starts and goals cannot be closer than 2 agent diameters
  proximity_limit = 4 * Agent.RADIUS

  for _ in range(num_agents):
    while True:
      start = np.random.uniform(low=-bound / 2.0, high=bound / 2.0, size=(1, 2))
      goal = np.random.uniform(low=-bound / 2.0, high=bound / 2.0, size=(1, 2))
      if not starts and not goals:
        # First pass, anything goes
        break
      # Check that the new start and goal aren't too close to anything else
      start_arr = np.full((len(starts), 2), start)
      goal_arr = np.full((len(goals), 2), goal)

      # TODO: We could probably get away without reconstructing these from scratch every iteration
      proposed_points = np.vstack((start_arr, start_arr, goal_arr, goal_arr))
      existing_points = np.vstack((*starts, *goals, *starts, *goals))
      if np.linalg.norm(proposed_points - existing_points, -np.inf) > proximity_limit:
        break

    # We've found a valid start and goal. Pick a random agent type, and build a transformer
    agent = ctor(start, goal, choice(agent_ctors))
    starts.append(start)
    goals.append(goal)
    agents[agent.get_id()] = agent

  # Output the starts and goals
  with open(scenario_file_path, 'w') as scenario_file:
    json.dump(
        {
            'bound': bound,
            'agents': {
                agent.get_id(): {
                    'start': agent.get_agent().pose.tolist(),
                    'goal': agent.get_agent().goal.tolist()
                } for agent in agents.values()
            }
        }, scenario_file)
  return agents


def run_simulation(num_agents=None,
                   scenario_file=None,
                   output_file=None,
                   use_transformers=True,
                   verbose=False):
  '''Main point of entry'''
  log = logging.getLogger(__name__)
  if verbose:
    coloredlogs.install(fmt=LOG_FMT, level=LOG_VERBOSE_LEVEL, field_styles=LOG_FIELD_STYLES)
  else:
    coloredlogs.install(fmt=LOG_FMT, level=LOG_BASE_LEVEL, field_styles=LOG_FIELD_STYLES)

  if scenario_file:
    log.info('Loading scenario from %s', scenario_file)
    transformers = load_scenario(scenario_file, use_transformers)
  elif num_agents:
    log.info('Generating %d agents with random starts and goals', num_agents)
    transformers = generate_agents(num_agents, use_transformers,
                                   f'scenario_{datetime.datetime.now().isoformat()}.json')
  else:
    log.critical(
        'Please provide either a number of agents to generate or a specific scenario file to run')
    sys.exit(-1)

  if output_file:
    output_path = output_file
  else:
    output_path = f'output_{datetime.datetime.now().isoformat()}.json'

  for tf_id in transformers:
    for other_tf_id in [other_id for other_id in transformers if other_id != tf_id]:
      transformers[tf_id].register_transformer(transformers[other_tf_id])

  agents = {id_k: tf.get_agent() for (id_k, tf) in transformers.items()}
  world = World(agents, transformers)
  for agent in agents.values():
    agent.set_world(world)

  log.info('Starting simulation')
  world.simulate()
  log.info('Simulation complete')
  log.info('Outputting trajectories to %s', output_path)
  world.output_logs(output_path)
  log.info('All done!')


if __name__ == '__main__':
  fire.Fire(run_simulation)
