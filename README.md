# VALUE ITERATION ALGORITHM

## AIM
Write the experiment AIM.

## PROBLEM STATEMENT
Explain the problem statement.

## POLICY ITERATION ALGORITHM
Include the steps involved in the value iteration algorithm

## VALUE ITERATION FUNCTION
```
REG NO:21221240028
NAME: MEENA S
```

```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
      Q=np.zeros((len(P),len(P[0])),dtype=np.float64)
      for s in range(len(P)):
        for a in range(len(P[s])):
          for prob,next_state,reward,done in P[s][a]:
            Q[s][a]+=prob*(reward+gamma*V[next_state]*(not done))
      if(np.max(np.abs(V-np.max(Q,axis=1))))<theta:
        break
      V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s , a in enumerate(np.argmax(Q,axis=1))}[s]
    return V, pi
```

## OUTPUT:
## Optimal Policy:
![269343953-a6fe8ca5-a428-42d0-a9ef-eb54d2bc03ee](https://github.com/MEENA155/rl-value-iteration/assets/94677128/b75c9afb-2286-47a0-b21b-6b8f1e63c6cf)

## Optimal Value Function:
![269344165-07a12d6e-173b-4099-b665-0cec64fbf38e](https://github.com/MEENA155/rl-value-iteration/assets/94677128/ae41dc1e-4845-43f9-b506-da34019652ea)

## Success Rate for Optimal Policy:
![269344373-e7df5ab6-47c8-4c57-97f9-4bd58b4ea95b](https://github.com/MEENA155/rl-value-iteration/assets/94677128/f1f472d3-cab2-4ee7-8c91-6b3e857cb3c2)

## RESULT:

Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.

