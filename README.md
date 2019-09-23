# Automatic Distributed Multi-Agent Coordination of Single-Agent Robot Controllers

## What's here

This repo contains a proof-of-concept implementation of the transformer protocol described in the
(under submission) paper "Automatic Distributed Multi-Agent Coordination of Single-Agent Robot
Controllers", as well as a simulator to run said implementation and a visualization generator for
the results of the simulator.

## How to use it
You'll need the following dependencies (all Python modules; install as you prefer): `coloredlogs`,
`icecream`, `verboselogs`, `numpy`, `fire`, `matplotlib`. This is Python 3 code.

You can run `python simulate.py --help` to see documentation on usage of the simulator; a good
example invocation is `python simulate.py --num-agents=5 --verbose` to generate a random scenario
with 5 agents and use verbose logging.

There's also `python visualize.py --help`; a corresponding example here might be `python
visualize.py scenario_foo.json output_bar.json --output-file=foobar.mp4` to load some scenario
`scenario_foo.json`, some simulation trace `output_bar.json`, and save the resulting visualization
to `foobar.mp4`.

## Warnings, etc.

This is research-grade code. As such, it comes with no support or guarantees, and is likely buggy.
Fixes for bugs will be added as they are discovered and as time permits.
