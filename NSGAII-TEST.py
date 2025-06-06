from NSGAII import *

def test(f, pop, cr, gen):
    y_vector = []
    for i in range(100):
        # f = 0.8 pop = 50 cr = 0.9
        pop1 = Populacao(f, 30, pop, cr, "bnk")
        pop1.comeback(5, 0.1, 2)
        pop1.run(gen)
        y_vector.append(pop1.pop[0])
    y_vector.sort(reverse=True)
    print("TEST: F: ", f, "- POP: ", pop, "- CR: ", cr, "MEDIAN: ", np.median(y_vector))
    return y_vector

fig, ax = plt.subplots()
now = datetime.datetime.now()
print("start time:", now)
start = time.time()
endtimek = 0
for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    for (pop, gen) in [(50, 100), (60, 84), (70, 71), (80, 63), (90, 56), (100, 50)]:
        for cr in [0.1, 0.3, 0.5, 0.7, 0.9]:
            ax.plot(test(f, pop, cr, gen), label="f: " + str(f) + " pop: " + str(pop) + " cr: " + str(cr))
            if endtimek == 0:
                end = time.time()
                duration = (end - start) * 270
                print("end time:  ", (now + datetime.timedelta(seconds=duration)))
                endtimek = 1



plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("best y value")
plt.title("Differential Evolution: Rosenbrock Function")
plt.legend()
plt.show()

