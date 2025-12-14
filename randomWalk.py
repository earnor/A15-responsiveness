seed(1)
pop = np.zeros((N,Y))
for s in scenList:
    popg = list()
    popg.append(0.005 if random() < 0.5 else 0.01)
    for i in range(1, Y):
        movement = -.0005 if random() < 0.48 else .001
        value = popg[i-1] + movement
        popg.append(value)
    pop[s,] = popg