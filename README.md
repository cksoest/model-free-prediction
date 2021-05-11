# model-free-prediction
## Tutorial Q-learning
```python
from grid import Grid
import numpy as np


rewards = np.array([
            [-1, -1, -1, 40],
            [-1, -1, -10, -10],
            [-1, -1, -1, -1],
            [10, -2, -1, -1]
        ])
policy = None
terminal_states = [(0, 3), (3, 0)]
gamma = 1

grid = Grid(rewards, policy, terminal_states, gamma)
grid.run_q_learning(500, 0.05, verbose=True)
```

## Beknopte uitleg code
Door middel van het aanmaken van een nieuw Grid worden de
attributen aangemaakt van het algoritme dat uitgevoerd word.

Vervolgens kan er met de volgende methode grid.run{naam_algoritme}
het desbetreffende algoritme worden uitgevoerd. In alle gevallen zou
dan meegegeven moeten worden om hoeveel iteraties het gaat, en in 
sommige gevallen moeten er extra parameters meegegeven worden zoals,
stapgrootte of epsilon.

Bij een policy evaluatie algoritme moet er ook een geldige policy
meegegeven worden. Dit is de policy die geëvalueerd word. In het 
voorbeeld hierboven staat de policy op None
omdat er voor dat algoritme geen policy nodig is.