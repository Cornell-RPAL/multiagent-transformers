'''Visualize the output of a simulation'''
import json
from functools import partial

import fire
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

from single_agent import Agent
from simulate import World

COLORS = [(87 / 255.0, 87 / 255.0, 87 / 255.0), (173 / 255.0, 35 / 255.0, 35 / 255.0),
          (42 / 255.0, 75 / 255.0, 215 / 255.0), (29 / 255.0, 105 / 255.0, 20 / 255.0),
          (129 / 255.0, 74 / 255.0, 25 / 255.0), (129 / 255.0, 38 / 255.0, 192 / 255.0),
          (129 / 255.0, 197 / 255.0, 122 / 255.0), (157 / 255.0, 175 / 255.0, 255 / 255.0),
          (41 / 255.0, 208 / 255.0, 208 / 255.0), (255 / 255.0, 146 / 255.0, 51 / 255.0),
          (255 / 255.0, 238 / 255.0, 51 / 255.0), (233 / 255.0, 222 / 255.0, 187 / 255.0),
          (255 / 255.0, 205 / 255.0, 243 / 255.0)]


def load_simulation(scenario_path, output_path):
  '''Parse the output and scenario of a simulation into a list of frames and metadata'''
  with open(scenario_path, 'r') as scenario_file:
    scenario_data = json.load(scenario_file)

  with open(output_path, 'r') as output_file:
    simulation_data = json.load(output_file)

  animation_data = {
      'bound': scenario_data['bound'],
      'agents': {
          int(agent_id): {
              'start': scenario_data['agents'][agent_id]['start'],
              'goal': scenario_data['agents'][agent_id]['goal'],
              'trace': []
          } for agent_id in scenario_data['agents']
      }
  }

  for frame in simulation_data:
    for agent_id in frame:
      animation_data['agents'][int(agent_id)]['trace'].append(frame[agent_id])

  return animation_data


def init(animation_data):
  '''Set up the shapes and figure for the animation'''
  fig = plt.figure()
  fig.set_dpi(800)
  fig.set_size_inches(10, 10)

  bound = animation_data['bound']
  bound += 3.0 * Agent.RADIUS
  ax = plt.axes(xlim=(-bound / 2.0, bound / 2.0), ylim=(-bound / 2.0, bound / 2.0))
  ax.axis('off')
  ax.axis([-bound / 2.0, bound / 2.0, -bound / 2.0, bound / 2.0])
  ax.set_aspect('equal')
  if len(animation_data['agents']) > len(COLORS):
    print('Warning! More agents than preset colors.')

  patches = {
      agent_id: plt.Circle((animation_data['agents'][agent_id]['start'][0][0],
                            animation_data['agents'][agent_id]['start'][0][1]),
                           Agent.RADIUS,
                           fc=COLORS[agent_id],
                           label=agent_id) for agent_id in animation_data['agents']
  }

  for agent_id in patches:
    start_center = (animation_data['agents'][agent_id]['start'][0][0],
                    animation_data['agents'][agent_id]['start'][0][1])
    start_patch = Wedge(start_center, Agent.RADIUS, 0, 360, width=0.02, fc=COLORS[agent_id])
    goal_center = (animation_data['agents'][agent_id]['goal'][0][0],
                   animation_data['agents'][agent_id]['goal'][0][1])
    goal_patch = Wedge(goal_center, Agent.RADIUS, 0, 360, width=0.02, fc=COLORS[agent_id])
    ax.annotate('S', xy=start_center, ha='center', va='center', fontsize=10)
    ax.annotate('G', xy=goal_center, ha='center', va='center', fontsize=10)
    circ = patches[agent_id]
    ax.add_patch(circ)
    ax.add_patch(start_patch)
    ax.add_patch(goal_patch)

  return fig, ax, patches


def animate(animation_data, patches, i):
  '''Generate each successive frame'''
  for agent_id in animation_data['agents']:
    pose = animation_data['agents'][agent_id]['trace'][i]['pose']
    x, y = pose[0]
    agent_circ = patches[agent_id]
    agent_circ.center = (x, y)

  return list(patches.values())


def visualize(scenario_file, simulation_file, output_file=None, display=True):
  '''Visualize a given simulation execution.
  scenario_file_path: The path to the JSON file with scenario parameters
  simulation_file_path: The path to the JSON file with the simulation log
  output_file_path: The (optional) path to the .mp4 file in which to save the visualization
  display: Whether or not to show the visualization'''
  animation_data = load_simulation(scenario_file, simulation_file)
  fig, ax, patches = init(animation_data)
  ax.legend(title='Agents')
  anim = animation.FuncAnimation(
      fig,
      partial(animate, animation_data, patches),
      frames=len(animation_data['agents'][0]['trace']),
      interval=World.TICK * 1000,
      blit=True)
  if display:
    plt.show()

  if output_file:
    anim.save(output_file, fps=30, dpi=100, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])


if __name__ == '__main__':
  fire.Fire(visualize)
