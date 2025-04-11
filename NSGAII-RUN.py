from NSGAII import *

def test(f, pop, cr, gen):
    y_vector = []
    for i in range(1):
        # f = 0.8 pop = 50 cr = 0.9
        pop1 = Populacao(f, 30, pop, cr, 0)
        pop1.comeback(5, 0.1, 2)
        pop1.run(gen)
        for j in range(len(pop1.pop)):
            print(pop1.pop[j].y, " ", pop1.pop[j].rank)
            y_vector.append(pop1.pop[j].y)
            if pop1.pop[j].rank > 0:
                break
    print("TEST: F: ", f, "- POP: ", pop, "- CR: ", cr, "MEDIAN: ")
    return y_vector

ys = test(0.8, 50, 0.9, 100)


fig, ax = plt.subplots()
x_vals, y_vals = zip(*ys)
ax.plot(x_vals, y_vals, color='blue', marker='o')
now = datetime.datetime.now()
print("start time:", now)
start = time.time()
endtimek = 0




plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("best y value")
plt.title("Differential Evolution: Rosenbrock Function")
plt.legend()
plt.show()

