'''Utility functions for collision testing'''

import numpy as np


def test_collision(a1, a2, r):
  '''Test if two actions collide at any point in their execution'''
  s1, v1, t1 = a1
  s2, v2, t2 = a2
  # First, handle the period where both are moving
  # The extrema will happen at the boundaries: start, ...
  d_0 = np.linalg.norm(s2 - s1)
  min_t = min(t1, t2)
  s1_t = s1 + v1 * min_t
  s2_t = s2 + v2 * min_t
  # ... and end
  d_t = np.linalg.norm(s2_t - s1_t)
  if d_0 <= 2 * r or d_t <= 2 * r:
    return True

  # Or in the middle, if the lines intersect
  v = (v2 - v1)
  if v.dot(v.T) != 0:
    t_coll = -v.dot((s2 - s1).T) / v.dot(v.T)
    if 0 <= t_coll <= min_t:
      s1_coll = s1 + v1 * t_coll
      s2_coll = s2 + v2 * t_coll
      d_coll = np.linalg.norm(s2_coll - s1_coll)
      if d_coll <= 2 * r:
        return True
  else:
    # If the denominator was zero, then the actions are parallel and if they weren't too close at
    # the start, they won't be too close ever
    return False

  # Now we need to handle the case where one is moving and the other is not
  actions = [a1, a2]
  min_i = np.argmin([t1, t2])
  first_stopped = actions[min_i]
  last_stopped = actions[(min_i + 1) % 2]

  s1, v1, t1 = first_stopped
  s2, v2, t2 = last_stopped

  # Advance the first-stopped action to its end
  s1 = s1 + v1 * t1

  # Find the min distance between the last-stopped action and the stopped point
  if v2.dot(v2.T) != 0:
    t_coll = v2.dot((s1 - s2).T) / v2.dot(v2.T)
    if min_t < t_coll <= t2:
      s2_coll = s2 + v2 * t_coll
      d_coll = np.linalg.norm(s2_coll - s1)
      if d_coll <= 2 * r:
        return True

  return False
