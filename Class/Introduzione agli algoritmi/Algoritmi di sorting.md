---
Created: 2024-04-22
Class: Introduzione agli algoritmi
Related: 
Completed:
---
---

>[!info] Index
>- [[#Teorema|Teorema]]
>- [[#Merge Sort|Merge Sort]]
>	- [[#Merge Sort#Funzionamento della funzione `Fondi()`|Funzionamento della funzione Fondi()]]
>	- [[#Merge Sort#Merge Sort iterativo|Merge Sort iterativo]]
>- [[#Quick Sort|Quick Sort]]
>	- [[#Quick Sort#Caso medio|Caso medio]]
>	- [[#Quick Sort#Quick Sort in-place|Quick Sort in-place]]
>- [[#Heap Sort|Heap Sort]]
>- [[#Counting Sort|Counting Sort]]
>- [[#Bucket Sort|Bucket Sort]]

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

> [!info]- Heap
> ![[Class/Introduzione agli algoritmi/Strutture dati#Heap]]

L'idea che sta dietro all'algoritmo è quella di:
1. organizzare gli elementi della lista da ordinare come heap minimo
2. estrarre gli elementi dall'heap uno alla volta e accodarli ad un nuovo vettore B inizialmente vuoto
3. restituire il vettore B

```Python
def Heapsort(A):
	import heapq
	heapify(A)                     # Θ(n)
	B = []
	while A:                       # Θ(n)
		B.append(heappop(A))       #    O(log n)
	return B
```

---
## Counting Sort
L’idea è quella di fare in modo che il valore di ogni elemento della sequenza determini direttamente la sua posizione nella sequenza ordinata
Il costo computazionale è di $\theta(n+k)$ (se $k=O(n)$ allora l’algoritmo ordina gli elementi in $\theta (n)$)

Questo algoritmo funziona come segue:
- trova `k`, l’elemento massimo dell’array `A` da ordinare
- inizializza l’array `C` con i `k` contatori delle occorrenze in A
- scorri `A` e per ogni indice `i` incrementa il contatore `C[A[i]]` delle occorrenze di `A[i]`
- scorri `C` e per ogni indice `i` inserisci `C[i]` occorrenze dell’elemento `i` in `A`

![[Screenshot 2024-04-22 alle 22.15.54.png|500]]

```python
def CountingSort(A):
	k = max(A)                              # Θ(n)
	n = len(A)
	# crea una lista C di contatori
	# per registrare il numero di
	# occorrenze degli elementi in A
	C=[0]*(k+1)                             # Θ(k)
	# calcola le occorrene degli
	# elementi in modo ordinato
	for i in range(n):                      # Θ(n)
		C[A[i]] += 1
	# reinserisce in A i suoi elementi
	# in modo ordinato
	j = 0
	for i in range(len(C)):                 # Θ(k+1)
		for _ in range(C[i]):               #    Θ(C[i])
			A[j] = i
			j += 1
```

---
## Bucket Sort
L’idea alla base di questo algoritmo di sorting è il dividere il vettore in *`k` sottointervalli* detti **bucket**, e distribuire i valori nei loro bucket per poi ordinarli separatamente. La posizione di ogni singolo elemento all’interno del bucket corrispondente è data dalla formula:
$$\left\lfloor  k\cdot\frac{x}{M+1}  \right\rfloor$$
Questa formula mi assicura che i numeri più piccoli andranno nei primi secchi mentre quelli più grandi negli ultimi, in modo tale da semplificare poi il lavoro alla macchina quando dovrà fare il vero e proprio sorting. In una situazione ideale dovrei avere lo stesso numero di elementi in ogni bucketzX

Il costo computazionale di questo algoritmo dipenderà dal sorting utilizzato per ordinare i vari buckets ma se gli elementi in input sono uniformemente distribuiti non ci si aspetta che molti elementi cadano nello stesso bucket (i.e. in ogni bucket ci saranno circa $\frac{n}{k}$ elementi ).

Questo algoritmo funziona come segue:
- crea una lista di `k` buckets inizialmente vuoti
- trova `M`, l’elemento massimo dell’array `A` da ordinare
- scorri `A` e per ogni valore `x` inserisci `x` nel bucket $B\left[ \left\lfloor  k \frac{x}{M+1}  \right\rfloor \right]$
- ordina gli elementi di ciascun bucket
- concatena gli elementi ordinati dei vari bucket

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

