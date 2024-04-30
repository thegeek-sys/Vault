---
Created: 2024-04-22
Class: Introduzione agli algoritmi
Related: 
Completed:
---
---

> [!info] Index
> - [[#Introduction|Introduction]]
> 	- [[#Introduction#Insiemi statici|Insiemi statici]]
> 	- [[#Introduction#Insiemi dinamici|Insiemi dinamici]]
> - [[#Heap|Heap]]
> 	- [[#Heap#Proprietà|Proprietà]]
> 	- [[#Heap#heapify(A)|heapify(A)]]
> 	- [[#Heap#heappop(A)|heappop(A)]]
> 	- [[#Heap#heappush(A, x)|heappush(A, x)]]
> - [[#Linked List|Linked List]]
> 	- [[#Linked List#Implementazione|Implementazione]]
> 		- [[#Implementazione#Esempio|Esempio]]
> 	- [[#Linked List#Creazione|Creazione]]
> 	- [[#Linked List#Aggiungere elementi|Aggiungere elementi]]
> 		- [[#Aggiungere elementi#Esercizio|Esercizio]]
> - [[#Pila e Coda|Pila e Coda]]
> 	- [[#Pila e Coda#Pila (stack)|Pila (stack)]]
> 	- [[#Pila e Coda#Coda (queue)|Coda (queue)]]
> 	- [[#Pila e Coda#Libreria Python|Libreria Python]]
> 

---
## Introduction
![[Pasted image 20240425112436.png]]

Una struttura dati è composta da:
- un **modo sistematico** di organizzare i dati
- un **insieme di operatori** che permettono di manipolare la struttura

Le strutture dati possono essere:
- **omogenee** → contiene tutti dati dello stesso tipo (ad esempio un array di interi)
- **disomogenee** → contiene dati di diversi tipi (ad esempio un dizionario che mette in corrispondenza interi e stringhe)

### Insiemi statici
Un insieme statico è una struttura dati omogenea in cui **la dimensione è fissa** e deve essere specificata al momento della creazione.
La dimensione non può cambiare durante l'esecuzione del programma e questi sono generalmente più efficienti in termini di spazio e accesso rispetto agli insiemi dinamici.

Un esempio di insieme statico è l'**array**.
```C
int A[50]
```

Python per accedere ad un elemento in una lista ci mette il doppio del tempo rispetto al tempo di accesso ad un elemento in un array, poiché quando si accede ad un elemento di una lista, eseguendo ad esempio il comando `A[50]` non si ha accesso direttamente all'elemento in posizione 50 ma all'indirizzo di memoria dell'elemento.

### Insiemi dinamici
Un insieme dinamico è una struttura dati in cui **la dimensione può cambiare** dinamicamente durante l'esecuzione del programma. Sono utili quando la dimensione dell'insieme è incerta o può cambiare nel tempo, quindi non si può specificare una dimensione massima in anticipo.
Gli insiemi dinamici offrono maggiore flessibilità ma possono richiedere **più memoria** e avere **prestazioni inferiori** rispetto agli insiemi statici, che sono più efficienti ma meno flessibili nella dimensione.
Un esempio di insiemi dinamici è la **lista a puntatori**.

Quindi si può concludere che non esiste una struttura migliore di un'altra, ma dipende dal problema che abbiamo bisogno di risolvere.

---
## Heap
La **struttura dati heap** in Python è implementata nella libreria `heapq`, questa libreria offre tre funzioni per gestire liste che fungono da heap:
- **Crea struttura**
	`heapify(A)` → trasforma una lista A in un heap 
- **Estrai minimo**
	`heappop(A)` → rimuove e restituisce l'elemento minimo della lista e ristabilisce le proprietà dell'heap
- **Inserisci elemento**
	`heappush(A, x)` → inserisce l'elemento x in modo che l'heap mantenga le sue proprietà

![[Heap.png|center|550]]

Le **proprietà** fondamentali della struttura dati heap sono:
- l'heap è un ordinamento verticale
- nell'heap l'**elemento minimo risiede nella radice** , quindi è il primo elemento della lista e può essere trovato in tempo costante
- ogni foglia è più piccola del suo ramo (genitore)

>[!hint]
>Si noti che scorrere il vettore da sinistra a destra corrisponde a muoversi sull’albero per livelli, dall’alto verso il basso e da sinistra a destra in ciascun livello
>Un ordinamento verticale garantisce anche l'ordinamento orizzontale, mentre il contrario non vale

Provando a vedere l’albero dell’heap come un ordinamento orizzontale ci è facile notare che:
- ogni nodo dell’albero binario corrisponde esattamente ad un elemento del vettore `A`
- la radice dell’albero corrisponde ad `A[0]`
- il figlio sinistro del nodo che corrisponde all’elemento `A[i]`, se esiste, corrisponde all’elemento `A[2i+1]`
- il figlio destro del nodo che corrisponde all’elemento `A[i]`, se esiste, corrisponde all’elemento `A[2i+2]`

### Proprietà
- Poiché lo heap ha tutti i livelli completamente pieni tranne al più l’ultimo, la sua altezza è $\theta(\log(n))$
- Con questa implementazione, la proprietà di ordinamento verticale implica che per tutti gli elementi tranne `A[0]` (poiché esso corrisponde alla radice dell’albero e quindi non ha genitore) vale: `A[parent(i)] ≤ A[i]`
- L’elemento minimo risiede nella radice, quindi può essere trovato in tempo $\theta(1)$

Ora vediamo le funzioni `heapify` e `heappop` nel dettaglio

### heapify(A)

```python
import heapq
A = [12, 3, 8, 4, 18, 16, 14, 11, 15, 17, 10, 1]
heapq.heapify(A)
print(A)  # [1, 3, 8, 4, 10, 12, 14, 11, 15, 17, 18, 16]
```

Attraverso la funzione `heapify` gli elementi della lista A vengono risistemati in modo da rispettare le proprietà dell'heap, discusse già precedentemente.
L’algoritmo `heapify` si avvale di una funzione ausiliaria `heapify1`, necessaria per il suo corretto funzionamento, ed ha lo scopo di mantenere le proprietà dell'heap.

![[Heapify.png]]
Opera sulla radice confrontandola con i suoi figli e se necessario, scambia la radice con il minore dei suoi figli, dopo che lo scambio è avvenuto, se non sono ancora rispettate le proprietà dell'heap la funzione si ripete ricorsivamente su quel nodo

```Python
def Heapify1(A, i):
	n = len(A)
	L = 2*i+1
	R = 2*i+2
	indice_min = i
	if L < n and A[L] < A[indice_min]:
		indice_min = L
	
	if R < n and A[R] < A[indice_min]:
		indice_min = R
	
	if indice_min != i:
		A[i], A[indice_min] = A[indice_min], A[i]
		Heapify1(A, indice_min)
	
```

### heappop(A)
L’idea è quella di salvare in una variabile `x` il minimo presente in `A[0]`. A questo punto copiamo in `A[0]` l’ultimo elemento dell’heap e chiamando la funzione `Heapify1` sulla radice dell’albero ricostruiamo così l’intero albero

```python
def Heapop(A):
	x = A[0]
	A[0]=A[len(A)-1]
	A.pop()
	Heapify1(A,0)        # O(log n)
	return x
```

### heappush(A, x)
Aggiungiamo l’elemento `x` all’ultimo posto dell’heap e facciamo risalire poi `x` nell’heap finché non risulta maggiore del padre o raggiunge la radice

```python
def Heappush(A, x):
	A.append(x)
	i=len(A)-1
	while i>0 and A[i]<A[(i-1)//2]:
		A[i],A[(i-1)//2]=A[(i-1)//2],A[i]
		i=(i-1)//2
```

---
## Linked List
Ogni elemento di lista è un record a due campi:
- **key** → è il campo che contiene l'informazione vera e propria
- **next** → contiene il puntatore che consente l'accesso all'elemento successivo (`None` in caso questo sia l’ultimo)

![[Record.png|center|400]]

se io voglio scorrere la lista devo scorrere ogni puntatore

La lista di Python è un ibrido tra la lista a puntatori e l'array. E’ simile alla lista a puntatori perché contiene puntatori, è simile ad un array perché consente l'indirizzamento diretto (es. `A[8]`)

### Implementazione
Una lista a puntatori è formata da **nodi**. Ogni nodo ha due aree, una destinata al dato, l'altra all'indirizzo che specifica dove si trova il nodo successivo

Implementazione della lista a puntatori in Python:
```Python
class Nodo:
	def __init__(self, key=None, next=None):
		self.key = key
		self.next = next
```
#### Esempio
![[Esempio1.drawio.png]]
```Python
p = Nodo(5)
print(P.key) # 5
print(P.next) # None

q = Nodo(6)
# ora devo collegare i due nodi
p.next = q
```


![[Esempio2.png]]
```Python
p = Nodo(7)
p.next = Nodo(2)
p.next.next = Nodo(12)
```

### Creazione
Funzione che data una lista `A`, crea una linked list
```Python
def crea(A):
	if A == []: return None # se la lista è vuota ritorno None
	
	p = Nodo(A[0])
	# creo un nuovo puntatore che utilizzerò per scorrere la lista
	# se restituisco direttamente q, questo non sarà
	# posizionato all'inizio, ma alla fine della lista
	p = q
	
	for i in range(1, len(A)):
		q.next = Nodo[A[i]]
		q = q.next
		
	return p
```

### Aggiungere elementi
Aggiungere un nuovo nodo in testa (all'inizio)
```Python
def es(p, x):
	return Nodo(x, p)
```
#### Esercizio
Creare una funzione che inserisca un nuovo elemento `x` dopo l’eventuale presenza del nodo `y`, se questo non è presente non inserire nulla, se sono presenti più occorrenze di `y` inserire alla prima occorrenza

```python
def es(p, x, y):
while p.next != None:
	if p.key == y:
		p.next == Nodo(x, p.next)
		break
	p = p.next
```

---
## Pila e Coda
### Pila (stack)
La **pila** è una struttura dati che esibisce un comportamento *LIFO* (Last In First Out). La pila può essere visualizzata come una pila di piatti: ne aggiungiamo uno appoggiandolo sopra quello in cima alla pila, e quando dobbiamo prenderne uno preleviamo quello più in alto.

Su una pila sono definite solo due operazioni
- **inserimento** → `push`
- **estrazione** → `pop`
Per fare in modo che queste uniche due operazioni siano efficienti è importante che esse abbiano complessità $\theta(1)$

```python
def push(Pila, x):
	Pila.append(x)         # Θ(1)

def pop(Pila):
	if Pila == []:
		return None
	return Pila.pop()      # Θ(1)
```

### Coda (queue)
La **coda** è una struttura dati che esibisce un comportamento *FIFO* (First In First Out). In altre parole, la coda ha la proprietà che gli elementi vengono da essa prelevati esattamente nello stesso ordine col quale vi sono stati inseriti.
La coda può essere visualizzata come una coda di persone in attesa ad uno sportello ed uno dei suoi più classici utilizzi è la gestione della coda di stampa, in cui documenti mandati in stampa prima vengono stampati prima.

Su una coda sono definite solo due operazioni:
- **inserimento** → `enqueue`
- **estrazione** → `dequeue`
Anche qui, per fare in modo che queste uniche due operazioni siano efficienti, è importante che esse abbiano complessità $\theta(1)$

```python
''' Implementazione tramite liste '''
def ins(Coda, x):
	Cosa.append(x)            # Θ(1)

def canc(Coda):
	if Coda == []:
		return None
	return Coda.pop(0)        # Θ(n)
```

Il problema di questa implementazione sta nel fatto che l’operazione di `canc` impiega $\theta(n)$. Per questo motivo ci conviene utilizzare una linked list con puntatore sulla testa e sulla coda

![[Screenshot 2024-04-25 alle 23.14.20.png]]

```python
def ins(testa, coda, x):
	p = Nodo(x)
	# se non è ancora stata creata la queue
	if coda == None:
		return p, p
	coda.next = p
	return testa, p

def canc(testa, coda):
	if testa == None:
		return None, None, None
	if testa == coda:
		return testa.key, None, None
	return testa.key, testa.next, coda
```

### Libreria Python
Dal modulo collections posso importare la struttura dati `deque` (*double ended queue*) che permette di effettuare inserzioni e cancellazioni da entrambi i lati di una lista in tempo $\theta(1)$, permettendoci quindi di implementare allo stesso tempo sia la **coda** che la **pila**

```python
from collections import deque

A = deque([])  # crea una lista vuota
A.pop()        # estrae l'ultimo elemento
A.append(x)    # inserisce x alla fine della queue
A.popleft()    # estrae il primo elemento dalla lista
A.appendleft() # inserisce x all’inizio in O(1)
```

---
## Alberi
