**note - in progress; I branched @tambetm's gist; I've fixed some issues and trying to get it closer to the paper. Pull request / help is welcome!!**

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

**Orginally based on:- **
https://gist.github.com/tambetm/0bd29b14d76b85946422b79f3a87df70 - https://gym.openai.com/evaluations/eval_sOUmkzSy26GIWJ5IIQeA#reproducibility
