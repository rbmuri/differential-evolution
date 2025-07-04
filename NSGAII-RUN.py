from NSGAII import *

chosen_function = "bnh"

def test(f, pop, cr, gen):
    y_vector = []
    for i in range(1):
        # f = 0.8 pop = 50 cr = 0.9
        pop1 = Populacao(f, 2, pop, cr, chosen_function)
        pop1.comeback(5, 0.1, 2)
        pop1.run(gen)
        for j in range(len(pop1.pop)):
            y_vector.append(pop1.pop[j].y)
            if pop1.pop[j].rank > 0:
                break
    print("TEST: F: ", f, "- POP: ", pop, "- CR: ", cr, "MEDIAN: ")
    if not os.path.exists(f"{chosen_function}_front.txt"):
        with open(f"{chosen_function}_front.txt", "w") as f:
            for row in pop1.pop:
                if row.rank == 0:
                    line = " ".join(map(str, row.y))
                    f.write(line + "\n")
    res = sorted(y_vector)
    return res

ys = test(0.8, 100, 0.9, 200)


fig, ax = plt.subplots()
x_vals, y_vals = zip(*ys)
ax.scatter(x_vals, y_vals, color='blue', marker='o', label='Pareto Points')
now = datetime.datetime.now()
print("start time:", now)
start = time.time()
endtimek = 0

n_eval()


#plt.yscale('log')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Pareto Optimal")
plt.legend()
plt.show()

