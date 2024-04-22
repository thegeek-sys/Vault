---
Created: 2024-04-22
Class: Introduzione agli algoritmi
Related: 
Completed:
---
---
## Teorema
Il costo computazionale di qualunque algoritmo di ordinamento basato su confronti è $\Omega(nlogn)$
Adesso ci verranno presentati tre algoritmi di ordinamento il cui caso pessimo richiede, infatti, un costo computazionale uguale a $\Theta(nlogn)$
questi sono:
- Merge Sort
- Quick Sort
- Heap Sort

---
## Merge Sort
Sia il Merge Sort che il QuickSort si basano sul paradigma **divide et impera**:
1. il problema viene suddiviso in sottoproblemi di dimensione inferiore (_divide_)
2. i sottoproblemi si risolvono ricorsivamente (_impera_)
3. le soluzioni dei sottoproblemi vengono fuse per ottenere la soluzione complessiva del problema

Nel Merge Sort infatti una sequenza di elementi viene divisa in due sottosequenze, queste vengono ordinate attraverso la ricorsione, che termina quando la sottosequenza è costituita da un solo elemento, allora le due sottosequenze vengono fuse in un'unica sequenza ordinata.

![[Screenshot 2024-04-22 alle 19.05.03.png|center|400]]

Il codice del Merge Sort può essere quindi riassunto in questo modo:
```Python
	def MergeSort(A, a, b):
		if a<b:
			m = (a+b)/2
			MergeSort(A, a, m)
			MergeSort(A, m+1, b)
			Fondi(A, a, m, b)
```

Da questo codice otteniamo questa equazione di ricorrenza:
$$
\begin{flalign}
T(n) = 2T(\frac{n}{2})+ \Theta(S(n))&&
\end{flalign}
$$
dove $S(n)$ rappresenta il costo di `Fondi()`, vediamo nel dettaglio il suo funzionamento

### Funzionamento della funzione `Fondi()`
- la funzione sfrutta il fatto che le sequenze sono ordinate
- il minimo della sequenza complessiva non può essere che il più piccolo tra i minimi delle due sottosequenze
- dopo aver eliminato da una delle due sottosequenze tale minimo, la proprietà rimane: il prossimo minimo non può essere che il più piccolo tra i minimi delle due parti rimanenti delle due sottosequenze

```Python
def Fondi(A, a, m, b):
	i, j = a, m+1
	B = []
	while i <= m and j <= b:      # O(n)
		if A[i] <= A[j]:
			B.append(A[i])
			i += 1
		else:
			B.append(A[j])        #    S(n)=Θ(n)
			j += 1
	while i <= m:
		B.append(A[i])            # O(n)
		i += 1
	while j <= b:                 # O(n)
		B.append(A[j])
		j += 1
	for k in range (len(B)):      # Θ(n)
		A[a+k] = B[k]
```

Il costo $S(n)$ della funzione `Fondi()` è $\Theta(n)$ quindi il costo computazione del Merge Sort sarà:
$$
T(n) = 2T\left(\frac{n}{2}\right)+ \Theta(n)
$$
portando così la complessità ad essere $T(n) = \Theta(nlogn)$

La fusione non si può effettuare direttamente in place all'interno di A poiché per fare spazio al minimo successivo sarebbe necessario spostare di una posizione tutta la sottosequenza rimanente per ogni nuovo minimo, ciò porterebbe la complessità ad essere $T(n) = \Theta(n^2)$

### Merge Sort iterativo
Esiste anche una versione iterativa di questo algoritmo
```python
def MergeSortI(A):
	n = len(A)
	l = 1
	while l < n:                         # Θ(log n)
		i = 0
		while i-l<n:                     # O(n/2l)
			Fondi(A, i, i+l-1, i+2*l-1)  #    O(l)
			i += 2*l
		l *= 2
```

Dunque si ha che:
$$
T(n) = \theta(\log(n))O(n) = \theta(n\log(n))
$$

---
## Quick Sort
L'algoritmo **Quick Sort** unisce in sé l'ordinamento in place del Selection Sort e il ridotto tempo di esecuzione del Merge Sort, mentre ha come svantaggio il fatto che nel caso peggiore la sua complessità sia $O(n^2)$
Tuttavia il suo tempo di esecuzione atteso come negli altri algoritmi di ordinamento basato sul confronto rimane $\Theta(n\log(n))$

Come anche detto in precedenza quick sort è un algoritmo che si basa sul **divide et impera**:
- **divide**
	nella sequenza di elementi seleziona un **pivot**. Il pivot viene selezionato in modo da ottenere due sottosequenze: quella degli elementi minori o uguali al pivot e quella degli elementi maggiori al pivot
- **impera**
	le due sottosequenze vengono ordinate ricorsivamente
- **passo base**
	la ricorsione finisce quando le sottosequenze sono costituite da un solo elemento

![[Screenshot 2024-04-22 alle 19.26.52.png|center|400]]

```Python
# non in-place
def QuickSort(A):
	if len(A) <= 1: return A
	pivot = A[0]
	L, M, R = [], [], []
	for i in A:
		if i < pivot: L.append(i)
		elif i == pivot: M.append(i)
		else: R.append(i)
	return QuickSort(L) + M + QuickSort(R)
```

In questo codice il pivot è il primo numero dell'array. Gli elementi dell'array vengono divisi in tre array più piccoli L, M, R, a seconda se siano rispettivamente più piccoli del pivot, uguali al pivot o più grandi del pivot. Alla fine le tre liste vengono concatenate per ottenere l'array ordinato.
Per quanto riguarda l’equazione di ricorrenza chiamiamo $k$ la dimensione del vettore `L` e si ha che $T(n)=T(k)+T(n+1-k)+\theta(n)$

Intuitivamente il **caso pessimo** si verifica quando il vettore è già ordinato (si avrà infatti che `len(S)==n-1`) e la ricorsione diventa $T(n)=T(n-1)+\theta(n)=\theta(n^2)$
Intuitivamente il **caso migliore** lo si ha quando, ad ogni passo, l’array viene diviso in due metà identiche tra `L` ed `R`. L’equazione di ricorrenza diventa quindi $T(n)=2(T\frac{n}{2}+\theta(n)=T(n\log(n)))$

### Caso medio
Utilizzando un quick sort randomizzato (il pivot viene ogni volta scelto casualmente) e ipotizzando la soluzione $n\log (n)$ si può dimostrare per sostituzione che il caso medio è $\mathbf{T(n)=\theta(n\log(n))}$

### Quick Sort in-place
 La versione del QuickSort che troviamo qui sopra non opera in place, vediamo ora lo pseudocodice di una implementazione che lavora in-place
 
```Python
def QuickSort(A, i, j)
	if i < j:
		p = Partition(A, i, j)
		QuickSort(A, i, p-1)
		QuickSort(A, p+1, j)
```
In questo codice la funzione `Partition` posiziona tutti gli elementi in modo che a sinistra del pivot ci siano solo numeri minori o uguali e a destra elementi maggiori o uguali. Ciò viene fatto senza usare ulteriore spazio.

```Python
def Partition(A, a, b):
	# il primo elemento viene selezionato come pivot
	pivot = A[a]
	# inizializzato il punto di divisione
	i = a+1
	for j in range(i, b+1):
		if A[j] < pivot:
			A[j], A[i] = A[i], A[j]
			i += 1
	A[a], A[i-1] = A[i-1], A[a]
	# resitutisce l'indice del pivot
	return i-1 
```
La funzione partiziona la porzione dell'array che va da a a b
1. Sceglie il primo elemento come pivot.
2. gli indici `j` e `k` vengono spostati verso destra a partire dalla destra del pivot.
3. ogni volta che l'indice `j` è su un elemento inferiore al pivot avviene uno scambio tra gli elementi puntati da `i` e `j`, successivamente `i` viene incrementato
4. quando `j` ha terminato di scorrere la porzione dell'array il pivot viene spostato nella sua posizione corretta ovvero `i-1`
5. l'indice del pivot viene restituito

---
## Heap Sort
L'Heap Sort è un algoritmo che ha un costo computazionale di $O(n\log(n))$ anche nel caso peggiore e come il Selection sort ordina in place.

Questo algoritmo sfrutta un'importante **struttura dati** che è l'**heap**, che, avendo delle specifiche proprietà, rappresenta la chiave per il corretto funzionamento dell'algoritmo.

### Heap: struttura dati
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

### Sorting
Ritorniamo ora all'implementazione dell'algoritmo di HeapSort
L'idea che sta dietro all'algoritmo è quella di:
1. organizzare gli elementi della lista da ordinare come heap minimo
2. estrarre gli elementi dall'heap uno alla volta e accodarli ad un nuovo vettore B inizialmente vuoto
3. restituire il vettore B

```Python
def Heapsort(A):
	from heapq import heapify, heappop
	heapify(A)
	B = []
	while A:
		B.append(heappop(A))
	return B
```

Come vediamo nell'implementazione di questo algoritmo sfruttiamo le funzioni **heapify** e **heappop**, ora le vediamo nel dettaglio

#### Funzionamento della funzione `Heapify()`

```
>>> import heapq
>>> A = [12, 3, 8, 4, 18, 16, 14, 11, 15, 17, 10, 1]
>>> heapq.heapify( A )
>>> print(A)
[1, 3, 8, 4, 10, 12, 14, 11, 15, 17, 18, 16]
```

Attraverso la funzione `heapify` gli elementi della lista A vengono risistemati in modo da rispettare le proprietà dell'heap, discusse già precedentemente.
Quindi ora i nodi presenti in ogni cammino radice foglia risultano ordinati in modo crescente e l’elemento minimo si troverà necessariamente al primo posto.

L’algoritmo di heapify si avvale di una funzione ausiliaria `heapify1`, necessaria per il suo corretto funzionamento, ed ha lo scopo di mantenere le proprietà dell'heap.

Opera sulla radice confrontandola con i suoi figli e se necessario, scambia la radice con il minore dei suoi figli, dopo che lo scambio è avvenuto, se non sono ancora rispettate le proprietà dell'heap la funzione si ripete ricorsivamente su quel nodo
![[Heapify.png]]


```Python
def Heapify1(A, i):
	n = len(A)
	L = 2*i +1
	R = 2*i +2
	indice min = i
	if L < n and A[L] < A[indice_min]:
		indice_min = L
	
	if R < n and A[R] < A[indice_min]:
		indice_min = R
	
	if indice_min != i:
		A[i], A[indice_min] = A[indice_min], A[i]
		Heapify1(A, indice_min)
	
```

## Bucket Sort

ordinamento che fa uso di secchielli

k numero di secchi che voglio usare
se i numeri sono n k è un numero <= n

ogni secchio corrisponde ad un intervallo, il numero di intervalli è  M/k

```Python
def (A, k):
	B = [[] for _ in range(k)]
	m = max(A)
	for x in A:
		i = x*k//(m + 1)
		B[i].append(x)
	for i in range(k):
		B[i].sort()
	C = []
	for i in range(k):
		C.extend(B[i])
	return C
```

il numero più grande finirà nell'ultimo secchio
`i` è un qualcuno numero compreso tra 0 e k-1

quando faccio il calcolo con la formula prendo la parte intera infatti nel codice ho //(m+1)


// foto 12:27 dell'esempio che ha fatto il prof farlo su drawio
il calcolo è (NUMx4)/10
4 è il numero di secchi nell'esempio
10 perchè 9+1 (m+1)

complessità
$$
T(n) = \Theta(n) +\sum\limits_{i=o}^{k-1} \text {costo di ordinare(B[i])}
$$

nel caso pessimo gli elementi non sono equidistribuiti, si concretizza quando in un vettore ci sono tutti elementi uguali, in tal modo tutti gli elementi finiscono nello stesso secchio, annullando di fatto l'utilità dell'algoritmo

