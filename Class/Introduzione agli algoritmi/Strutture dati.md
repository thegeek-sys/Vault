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
> - [[#Alberi|Alberi]]
> 	- [[#Alberi#Alberi radicati|Alberi radicati]]
> 	- [[#Alberi#Alberi binari|Alberi binari]]
> 	- [[#Alberi#Rappresentazione di alberi|Rappresentazione di alberi]]
> 		- [[#Rappresentazione di alberi#Tramite puntatori|Tramite puntatori]]
> 			- [[#Tramite puntatori#Esempio|Esempio]]

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
def es(P, x):
	P = Nodo(x, P)
	return P
```

Aggiungere un nuovo nodo in coda (alla fine)
```Python
def es(P, x):
	P.next = Nodo(x)
	return P
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
L’**albero** è una struttura dati estremamente versatile, utile per modellare una grande quantità di situazioni reali e progettare le relative soluzioni algoritmiche.

### Alberi radicati
Gli alberi radicati rappresentano il tipo più generico di albero in cui si ha che:
- i nodi sono organizzati in **livelli** numerati in ordine crescente allontanandosi dalla radice (di norma posta al livello zero)
- l’**altezza** di un albero radicato è la lunghezza del cammino più lungo dalla radice ad una foglia che è pari a $\theta(\log(n))$
- dato qualunque nodo $v$ che non sia la radice, il primo nodo che si incontra sul cammino da $v$ alla radice viene detto **padre di v**
- nodi che hanno lo stesso padre sono detti **fratelli** e la radice è l’unico nodo che non ha padre
- ogni nodo sul cammino da $v$ alla radice viene detto antenato di $v$
- tutti i nodi che ammettono $v$ come padre sono detti **figli di v**, ed i nodi che non hanno figli sono detti **foglie**
- tutti i nodi che ammettono come antenato vengono detti discendenti di v

![[Screenshot 2024-04-30 alle 17.00.54.png|600]]

Un albero radicato si dice **ordinato** se attribuiamo un qualche ordine ai figli di ciascun nodo, nel senso che se un nodo ha figli, allora vi è un figlio che viene considerato primo, uno che viene considerato secondo, etc.

Una particolare sottoclasse di alberi radicati e ordinati è quella degli **alberi binari**, che hanno la particolarità che ogni nodo ha al più due figli. Poiché sono alberi ordinati, i due figli di ciascun nodo si distinguono in **figlio sinistro** e **figlio destro**

### Alberi binari
Un albero è detto **binario** se ogni nodo al più possiede due figli.
Un albero inoltre è detto **completo** se possiede tutte le foglie sul medesimo livello (quasi completo se invece tutti i livelli tranne l’ultimo contengono il massimo numero possibile di nodi mentre l’ultimo livello è riempito completamente da sinistra verso destra solo fino ad un certo punto)

![[Screenshot 2024-04-30 alle 17.49.13.png|600]]

il numero di foglie è $2^h$
il numero dei nodi interni è $\sum^{h-1}_{i=0}2^i=2^h-1$
il numero totale dei nodi è $2^h+2^h-1=2^{h+1}-1$

Un albero completo si dice **bilanciato** in quando il rapporto tra il numero di nodi e l’altezza è esattamente $\log(n)$

### Rappresentazione di alberi
#### Tramite puntatori
Il modo più naturale di rappresentare e gestire gli alberi binari è per mezzo dei puntatori. Ogni singolo nodo è costituito da un record contenente:

![[Screenshot 2024-04-30 alle 18.04.24.png|center|150]]
- **key** → le opportune informazioni pertinenti al nodo stesso
- **left** → il puntatore al figlio sinistro (oppure `None` se il nodo non ha figlio sinistro)
- **right** → il puntatore al figlio destro (oppure `None` se il nodo non ha figlio destro)
All’albero si accede grazie al puntatore alla radice

```python
Class NodoAB:
	def __init__(self, key, left, right):
		self.key = key;
		self.left = left;
		self.right = right;
```

##### Esempio
![[Immagine 30-04-24 - 18.16.jpg|150]]
```python
>>> r = NodoAB(4)
>>> r.left = NodoAB(5)
>>> r.left.right = NodoAB(0)
>>> r.right = NodoAb(8)
```

---
## Dizionari
Un dizionario è una struttura dati che permette di gestire un insieme dinamico di dati, che di norma è un insieme totalmente ordinato, tramite queste tre sole operazioni:
- `insert` → si inserisce un elemento
- `search` → si ricerca un elemento
- `delete` → si elimina un elemento

Questa particolare struttura dati è basata **tabella hash**, una struttura dati molto efficiente che, sotto  ragionevoli assunzioni, riesce ad implementare le tre operazioni in **complessità media costante** $\Theta(1)$
Qui di seguito ne illustreremo due diverse implementazioni:
- tabella hash **chiusa**
- tabella hash **aperta**
L’utilizzo dell’indirizzamento diretto è preferibile alla tabella hash nel caso in cui l’universo delle chiavi è ragionevolmente piccolo ed il numero $n$ di elementi da memorizzare è vicino al numero $m$ delle possibili chiavi.

Dimensionare la tabella in base al numero di elementi **attesi** (che indicheremo con $m$ ed utilizzare una speciale funzione (funzione hash) per indicizzare la tabella
Una **funzione hash** è una funzione che data una chiave $x$ restituisce la posizione della tabella in cui l’elemento con chiave $x$ viene memorizzato:
$$
h(x)\in\{0,1,\dots,m-1\}
$$
la dimensione della tabella può non coincidere con la dimensione dell’universo, anzi in generale $m < < |U|$


![[Screenshot 2024-05-13 alle 22.55.58.png|center|400]]
L’idea è quella di definire una funzione d’accesso che permetta di ottenere la posizione di un elemento data la sua chiave, introducendo però il fenomeno delle **collisioni**

### Collisioni
Il verificarsi di collisioni è inevitabile quando l’insieme $U$ dei valori che le chiavi possono assumere è molto grande e la cardinalità $m$ degli indici disponibili è invece molto più piccolo di $U$.

>[!info]
>anche se le chiavi da memorizzare sono meno di $m$, non si può escludere che due chiavi $x\neq y$ siano tali per cui $h(x)=h(y)$ ossia che la funzione hash restituisca lo stesso valore per entrambe le chiavi, che quindi andrebbero memorizzate nella stessa posizione della tabella.

Stando così le cose le **collisioni**, vanno **evitate** il più possibile e, altrimenti, risolte.

### Funzione hash
Le funzioni hash sono dunque funzioni matematiche che prendono in input un dato (come una stringa o un numero) e producono in output un valore hash, che è tipicamente un numero intero che fa **riferimento ad un indirizzo all'interno di una tabella** (tabella hash). 

L'obiettivo principale di una funzione hash è quello di **distribuire uniformemente i dati** in modo casuale all'interno di un range di valori possibili onde evitare il più possibile il verificarsi di collisioni.

Supponiamo di avere una tabella di dimensioni in cui vogliamo inserire record le cui chiavi siano stringhe. Una possibile funzione hash in questo caso sarebbe:
```python
def hash_code(s, size):
	# size -> dimensione tabella
	# funzione hash per stringe
	h = 0
	for c in s:
		h += ord(c) # somma il valore ASCII di tutti i char
	return h % size

>>> hash_code('Angelo Monti', 20) # 9
```
la funzione `hash_code()` prende una stringa in input e calcola il suo hash sommando i valori ASCII di tutti i caratteri della stringa e poi calcolando il resto della divisione con la dimensione della tabella (quest'ultima operazione assicura che il valore restituito sia un indice all'interno della tabella).
Questa soluzione però può portare a collisioni, la distribuzione non sarebbe uniforme e inoltre non è efficiente (dipende dalla lunghezza delle stringhe)

La situazione ideale è quella in cui ciascuna delle $m$ posizioni della tabella è scelta con la stessa probabilità, ipotesi che viene detta **uniformità semplice della funzione hash**.


### Indirizzamento chiuso (liste di trabocco)
In questo caso ad ogni cella della tabella hash si fa corrispondere invece di un elemento, una `Lista` (solitamente una lista concatenata) detta **lista di trabocco**. In questo modo un elemento che collide viene aggiunto alla lista corrispondente all'indice ottenuto