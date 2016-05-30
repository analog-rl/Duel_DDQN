## note - in progress; i found some bugs in @tambetm's code

Used dueling network architecture with Q-learning, as outlined in this paper:

**Dueling Network Architectures for Deep Reinforcement Learning**  
*Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas*  
http://arxiv.org/abs/1511.06581

Command line:
```
# python duel.py CartPole-v0 --gamma 0.995 # note - i will double check this once finish debuing
# 
```

from code:
```
import sys
sys.argv = ['duel.py']
sys.argv += ['CartPole-v0']
sys.argv += ['--gamma', '0.995']
sys.argv += ['--no_display']
sys.argv += ['--gym_record', 'th']
```
