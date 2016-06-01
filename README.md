**note - in progress; i found some bugs in @tambetm's code**

Used dueling network architecture with Q-learning, as outlined in this paper:

**Dueling Network Architectures for Deep Reinforcement Learning**  
*Ziyu Wang, Tom Schaul, Matteo Hessel, Hado van Hasselt, Marc Lanctot, Nando de Freitas*  
http://arxiv.org/abs/1511.06581

Command line:
```
python duel.py CartPole-v0 --gamma 0.995 --no_display --batch_size 100 --gym_record th
 
```

from code:
```
import sys
sys.argv = ['duel.py']
sys.argv += ['CartPole-v0']
sys.argv += ['--gamma', '0.995']
sys.argv += ['--no_display']
sys.argv += ['--batch_size', '100']
sys.argv += ['--gym_record', 'th']
sys.argv += ['--verbose', '1']
```

**Change list / Todo**
- [x] fix - train on target_model (was on model)
- [x] fix - memory buffer is infinite
- [x] fix - DDQN: Updating weights was happening every frame
- [ ] ?? - investigate if batch should be every frame (or every X timestep)
- [x] remove HACK - batch_size is 10 (just pass a parameter)
- [ ] remove HACK - memory buffer size is hard coded 
- [ ] remove HACK - DDQN: Updating weights update interval is hardcoded
