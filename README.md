### sabemos: 
algoritmos F, CR é bom

### não sabemos: 
qual é melhor _(leandro e henrique?)_
- não adaptar N (provavelmente a pior)
- mudar deterministicamente
- adaptar comparando duas populações
    - N, 2N (10, 20) (15, 30) (20, 40)
    - N, N+10  
- **estagnação** _(sasawork?)_
    - minimo local
        - aumentar a população
    - não minimo local
        - mexer F (?)

### fazer agora
guardar os melhores x das ultimas 5 ou 10 geracoes e ter 10% de chance de, na hora de mutar novos 
individuos, escolher o x3 (x4-x3) das ultimas geracoes
- teste 1: mexer so no x3 ou em todos os x1 x2 x3 x4
- teste 2: 5 geracoes ou 10
- teste 3: melhor de cada geracao ou 5 melhores
- teste 4: 10% ou 5% ou outros
- teste novo: (x4-x3) comparar o x4 com o x3 e trocar se x4 for menor que 

se o algoritmo estiver indo lento, aumentar a diversidade / tamanho da populacao. se estiver indo rapido, diminuir a populacao para economizar

VALORES RECOMENDADOS
pop = 5*dim
f = random num 0-1
CR = random num 0-1 (?)

### fazer 

operadores mutação e crossover


### COMO IMPLEMENTAR ALGORITMO NO PYMOO

1. Fazer com que sua classe de algoritmo herde de pymoo.core.core.Algorithm

2. Reimplementar "__init__" de forma que herde da classe pymoo.core.core.Algorithm

3. Reimplementar "infill()" (retorno das soluções candidatas para serem avaliadas)
de forma que herde da classe pymoo.core.core.Algorithm

4. Reimplementar "advance()" (seleção) de forma que herde da classe pymoo.core.core.Algorithm

5. Reimplementar "has_next()" (decisão de parada) de forma que herde da classe 
pymoo.core.core.Algorithm

6. Adaptar meus dados de entrada e saída para corresponder com as estruturas de dados do Pymoo

# SBX -> simulated binary crossover!
# mutação polinomial 



