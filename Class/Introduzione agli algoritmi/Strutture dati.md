---
Created: 2024-04-22
Class: Introduzione agli algoritmi
Related: 
Completed:
---
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
