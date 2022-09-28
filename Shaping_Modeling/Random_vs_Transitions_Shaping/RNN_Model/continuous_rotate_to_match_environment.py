
from gym import spaces

observation_space = spaces.Dict(spaces={'target_trap_orientation':
                                             spaces.Discrete(2),
                                         'manipulandum_angle':
                                             spaces.Box(0, 359, dtype=float)})

action_space = spaces.Discrete(3)

