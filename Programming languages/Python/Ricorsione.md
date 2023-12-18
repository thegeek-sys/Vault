---
Created: 2023-11-29
Programming language: "[[Python]]"
Related: 
Completed:
---
---
## Introduction
Una funzione si dice **ricorsiva** quando nella sua definizione la funzione chiama sé stessa. All’interno di essa però deve anche essere presente un **caso base**
[[recurs.py|File ricorsivo]]

---
## Requisiti per risolvere un problema con ricorsione
1) `[Riduzione]` Verificare se è possibile ricondursi a problemi più semplici partendo da quello iniziale
2) `[Caso base]` Deve esistere almeno un problema con una soluzione elementare
3) `[Convergenza]` Deve essere sempre possibile, applicando la riduzione, di arrivare ad un caso base
4) `[Conquer]` È necessario unire le soluzioni delle riduzioni per risolvere il problema principale

## Queue e stack
Innanzitutto è importante fare distinzione tra e *Queue* e *Stack*

```start-multi-column
ID: ID_v5fv
Number of Columns: 2
Column Size: [43%, 56%]
```

### Queue
![[queue.png]]

Nella queue vige la regola **first in, last out**

--- column-end ---

### Stack
![[stack.png]]

Nello stack vige la regola **first in, first out**
--- end-multi-column
#### Stack di un programma
![[stack program.png]]

---
## Sequenza Fibonacci - ricorsivo
```python
def fibonacci(n):
	if n < 2:
		return 1
	else:
		return fibonacci(n-1) + fibonacci(n-2)
```

Una rappresentazione sotto forma di albero binario delle chiamate nello stack
![[fibonacci.png]]

**Ma cosa  succede  se tolgo il  caso base?** Succede che incorrerò in un `RecursionError`
Il problema di questo tipo di ricorsione è che spesso vengono ricalcolati più volte gli stessi valori tanto che questo algoritmo naive ha complessità $T(n) = T(n-1) + T(n-2) + \Theta(n)$

---
## Ricorsione con memorizzazione
La ricorsione con memorizzazione è anche detta **caching**. Infatti nel caso in cui devo lavorare su numeri molto grandi tramite ricorsione mi conviene salvare (memorizzare) i valori già calcolati piuttosto che ricalcolarli (per esempio in Fibonacci non mi serve ricalcolare più volte $f(2)$ ) in modo tale da poter  tagliare di molto la complessità di un programma ricorsivo.
Questa implementazione mi è possibile attraverso un dizionario a cui appendo ogni nuovo valore di $n$ che viene calcolato in modo tale che ogni volta che esso mi serve mi basta fare una query al dizionario portando la complessità del programma da esponenziale a lineare $\Theta(n-1)$ (il numero di chiamate ricorsive senza memorizzazione è di 88 mentre nel programma con memorizzazione è solamente di 9).

```python
def fibonacci_memo(n):
	'''
	Fibonacci with recursion + memorization
	'''
	memory = {0: 1, 1: 1} # caso base direttamente in memoria
	
	if n in memory:
		return memory[n]
	else:
		rez = fibonacci(n-1) + fibonacci(n-2)
		memory[n] = rez
		return rez

```

**Memorization** vuol dire ignorare le chiamate ricorsive e semplicemente accedere alla memoria in tempo costante; o in altri termini aggiungere dei nuovi casi base. Aumentando i casi base si velocizza la ricorsione

---
## Iterativo vs ricorsivo
**Su carta**, versione iterativa (bottom up) e ricorsiva (top down, con memorization) hanno la stessa complessità
**Sul calcolatore**, la versione iterativa può performare meglio perché evita di aprire e chiudere funzioni su stack del programma, ma in generale possiamo dire che la ricorsione potrebbe risultare più intuitiva da scrivere se localizziamo i sotto problemi

---
## Ricorsione all’andata
Possiamo anche fare in modo che il calcolo della ricorsione venga fatto all’andata fino a che non si arriva al caso base invece di calcolarlo al ritorno (ma il codice ci verrà molto probabilmente più complesso).

```python
# sommo da 1...N
''' RITORNO '''
# 1. incremento i -> i+1
# 2. finisco quando i==n+1 (convergenza e risultato)
# 3. in partenza la somma e' 0, ad ogni passo incremento
def sumrp(i, n, partial_sum=0):
    # 2. convergenza e risultato
    if i == n+1:
        return partial_sum # torniamo il caso generato
    # incremento della soluzione per ogni passo
    # sono ad iterazione i+1 e accumula la somma parziale
    return sumrp(i+1, n, partial_sum=partial_sum+i)

''' ANDATA '''
def sumrp(i, n, partial_sum=0):
    # 2. convergenza e risultato
    if i == n:
        return partial_sum + n # mi risparmio un passo ricorsivo
    return sumrp(i+1, n, partial_sum=partial_sum+i)
```

---
## Esercizio ricorsivo esame
Consideriamo questo gioco: dato lo stato della lista L calcolare tutti gli altri stati considerando come mossa la seguente proprietà:
- due elementi consecutivi devono avere il solito resto se divisi per due (**mossa**)
prossimo stato:
- quando la mossa **si verifica**
- allora si crea un nuovo stato L'
- che sostituisce agli elementi consecuitivi 
- la somma dei due consecutivi (riduzione)
Enumerare tutti i possibili stati e tornare tutte le foglie dell'albero di gioco

```python
state = [99, 1, 3, 5, 20]

def game(state, L=None):
	# mi salvo se sono nella prima chiamata
	start = False
	if L is None:
		# lo sono
		start = True
		# init una lista vouta
		L = []
	# definisco booleano per foglia
	leaf = True
	# provo le mosse
	for i in range(len(state)-1):
		pre, post = state[i, i+2]
		# se la mossa è verificata almeno una volta
		if pre % 2 == post % 2:
			# ricorsione, NON sono in una foglia
			leaf = False
			somma = pre + post
			# nuovo stato tutti tranne i e i+1 ma metto la somma
			state_next = state[:i]+[somma]+state[i+2:]
			# ricorsione su next state
			game(state_next, L)
		
	# se dato uno stato, non entriamo mai in ricorsione allora foglia
	if leaf:
		L.append((state, [ 'd' if s%2 == 1 else 'p' for s in state ]))
	# se era la prima chiamata torno L
	if start: return L

out = game(state)
print(out)

```