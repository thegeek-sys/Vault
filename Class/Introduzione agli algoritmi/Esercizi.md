---
Created: 2024-05-01
Class: Introduzione agli algoritmi
Related: 
Completed:
---
---
# ES.1
Dati i due puntatori a due liste concatenate di n interi progettare un algoritmo che in tempo $\theta(n^2)$ restituisca una terza lista contenente l'intersezione delle prime due (vale a dire i nodi con le chiavi comuni ad entrambe le liste di partenza).
Ogni chiave deve comparire una sola volta nella lista restituita e non importa l'ordine con cui le chiavi compaiono nella lista.

```python
def trovato(P, x):
	while P != None:
		if P.key == x: return True
		P = P.next
	return False

def es(A, B):
	O = None
	while A != None:
		if trovato(B, A.key) and !trovato(O, A.key):
			O = Nodo(x, O)
		A = A.next
	return O
```

---
# ES.2
## 1.
Data una lista tramite il puntatore al suo primo elemento, restituire il puntatore all’ultimo elemento se la lista ha almeno un elemento, `None` altrimenti. Calcolarne il tempo di esecuzione

```python
def es(A):
	while A != None:
		if A.next == None:
			return A
		A = A.next
	return None
# Θ(n)
```
## 2.
Data una lista tramite il puntatore al suo primo elemento, restituire il puntatore al penultimo elemento se la lista ha almeno due elementi, `None` altrimenti. Calcolarne il tempo di esecuzione
```python
def es(A):
	while A != None:
		if A.next.next == None:
			return A
		A = A.next
	return None
# Θ(n)
```
## 3.
Data una lista tramite il puntatore al suo primo elemento, restituire il puntatore alla stessa lista da cui sia stato eliminato l’ultimo elemento. Calcolarne il tempo di esecuzione
```python
def es(A):
	P = A
	while P != None:
		if P.next == None:
			P = None
		P = P.next
	return A
# Θ(n)
```
## 4.
Data una lista tramite il puntatore al suo primo elemento, restituire il puntatore di una lista che contenga gli stessi record della lista di partenza ma in ordine inverso (N.B. non deve essere creato alcun record, ma bisogna “smontare” e “rimontare” opportunamente i record iniziali)
```python
def es(A):
	P = None
	while A != None:
		P = Nodo(A.key, P)
		A = A.next
	return P
# Θ(n)
```

---
# ES.3
## 1.
Data una lista tramite il puntatore al suo primo elemento, restituire i puntatori a due liste, una con gli elementi di posto pari nella lista di partenza, ed una con gli elementi di posto dispari (anche qui, non bisogna creare nuovi nodi)
```python
def es(A):
	P = None
	D = None
	i = 0
	while A != None:
		if i%2 == 0:
			P = Nodo(A.key, P)
		else:
			D = Nodo(A.key, D)
		i += 1
	return P, D
```
## 2.
Data una lista di interi tramite il puntatore al suo primo elemento, stampare tutti i valori che compaiono almeno due volte nella lista
```python
def es(A):
	occ = dict()
	while A != None:
		occ[A.key] = occ.get(A.key, 0) + 1
		A = A.next
	
	for k, v in occ.items():
		if v > 1:
			print(k)
```
## 3.
Data una lista ordinata di interi tramite il puntatore al suo primo elemento ed un intero `x`, aggiungere `x` alla lista in modo da rispettare l’ordinamento
```python
def es(A, x):
	P = A
	x = Nodo(x)
	while P != None:
		if P.key < x.key:
			x.next = P.next
			P.next = x
			return A
		P = P.next
```
## 4.
Data in input una lista di interi tramite il puntatore al primo elemento, restituire la lista ordinata (senza creare nuovi nodi)
```python

```

---
# ES.4
## 1.
Dato il puntatore ad una lista doppiamente puntata restituire la lunghezza della lista
```python
def es(A):
	i = 0
	while A != None:
		i += 1
		A = A.next
	return i
```
## 2.
Dato il puntatore ad una lista doppiamente puntata stampare gli elementi della lista in ordine inverso (i.e. dall’ultimo al primo)
```python
def es(A):
	while A.next != None:
		A = A.next
	while A != None
		print(A.key)
		P = P.prev
```
## 3.
Dato il puntatore ad una lista di interi doppiamente puntata ed un intero `x` cancellare tutte le occorrenze di `x` dalla lista !!!CONTROLLARE
```python
def es(A, x):
	P = A
	while P != None:
		if P.key == x:
			P = P.prev
			P.next = P.next.next
		P = P.next
	return A
```
## 4.
Dato un intero `x` ed il puntatore al primo elemento di una lista di interi ordinata e doppiamente puntata inserire l’intero `x` nella lista in modo da mantenere l’ordine
```python
def es(A, x):
	P = A
	x = Nodo(x)
	while P != None:
		if P.key < x.key:
			x.next = P.next
			P.next = x
			return A
		P = P.next
```
## 5.
Dato il puntatore ad una lista di interi doppiamente puntata ed ordinata cancellare dalla lista tutti i duplicati
```python
def es(A):
	P = A
	D = set()
	while P != None:
		if P.key not in D:
			D.add(P.key)
		else:
			P.prev = P.next.next
			P = P.prev
		P = P.next
	return A
```

---
# ES.5
Scrivere una funzione `creaAlbero(n, m)` che genera un albero a caso con $n$ nodi aventi chiavi casuali nell’intervallo $[1,\dots ,m]$

```python
def creaAlbero(n, m):
	if n==0 return None
	p = NodoAB(random.randint(1, m))
	fs = random.randint(0, n-1)
	p.left = creaAlbero(fs, m)
	p.right = creaAlbero(n-fs-1, m)
	return p
```

---
# ES.6
Scrivere una funzione `stampa(r)` che stampa le chiavi dei nodi dell’albero `r`. Stampa ricorsivamente in base a queste regole: prima la chiave del nodo e poi, indentate, le chiavi del sottoalbero di sinistra e quelle del sottoalbero di destra

```python
def stampaAlbero(p, h=0):
	if p==None:
		print('| '*h,'-')
	else:
		print('| '*h, p.key)
		stampaAlbero(p.left, h+1)
		stampaAlbero(p.right, h+1)
```

---
# ES.7
Scrivere una funzione ricorsiva che dato in input un puntatore alla radice di un albero binario `p` e restituisca il numero di nodi totale in $\theta(n)$

```python
def es(p):
	if p==None: return 0
	s = es(p.left)
	d = es(p.right)
	return s + d + 1
```

---
# ES.8
Scrivere una funzione ricorsiva che prenda in input un puntatore alla radice di un albero binario `p` ed un elemento `x` e verifichi la presenza o meno dell’elemento nell’albero

```python
def es(p, x):
	if p==None: return False
	if p.key == x: return True
	if es(p.left, x): return True
	es(p.right, x)
```

# ES.9
Scrivere una funzione ricorsiva che prenda in input un puntatore alla radice di un albero binario `p` e ritorni l’altezza dell’albero

```python
def es(p):
	if p==None: return -1
	hs = es(p.left)
	hd = es(p.right)
	return max(hs, hd) + 1
	
```

---
# ES.10
Scrivere una funzione ricorsiva che prenda in input un puntatore alla radice di un albero binario `p` e un livello `h` ritorni il numero di nodi presenti al livello `h`

```python
def es(p, h):
	if p==None: return 0
	if h==0: return 1
	ls = es(p.left, h-1)
	ld = es(p.right, h-1)
	return ls + ld
```

---
# ES.11
In un array ordinato `A` di $n$ interi compaiono tutti gli interi da $0$ ad $n−2$. Esiste dunque nell’array un unico elemento duplicato. Si progetti un algoritmo iterativo che, dato `A`, in tempo $\theta(\log n)$ restituisca l’elemento duplicato.
Ad esempio, per `A=[0,1,2,3,4,4,5,6,7]` l’algoritmo deve restituire 4.

```python
def es(A):
	i, j = 0, len(A)-1
	while i<j:
		m = (i+j)//2
		if A[m] == m:
			i = m+1
		else:
			j=m
	return A[i]
```

---
# ES.12
Dato un array `A` di $n$ interi compresi tra $0$ a $50$, sapendo che nell’array sono certamente presenti dei duplicati, si vuole determinare la distanza massima tra le posizioni di due elementi duplicati in `A`
Ad esempio per `A=[3,3,4,6,6,3,5,5,5,6,6,9,9,1]` i soli elementi che in A si ripetono sono 3, 6 e 9.
- La distanza massima tra duplicati del 3 è 5,
- la distanza massima tra duplicati del 6 è 7,
- la distanza massima tra duplicati del 9 è 1.
quindi la risposta per l’array `A` è 7.
Progettare un algoritmo che, dato `A`, in tempo $\theta(n)$ restituisca la distanza massima tra le posizioni con elementi duplicati

```python
def es(A):
	C = [-1]*51
	m = 0
	for i in range(len(A)):
		if C[A[i]] == -1:
			C[A[i]] == i
		else:
			m = max(m, i-C[A[i]])
	return m
```

---
# ES.14
Dato un puntatore ad un albero binario di ricerca restituirlo sotto forma di vettore
![[59F7008D-4E70-4362-A492-746DFD3CF585.jpeg]]

```python
def es(p):
	if p==None: return []
	h = altezza(p)
	A=[None]*(2**(h+1))
	inserisci(p, A)
	return A

def altezza(p):
	if p==None: return -1
	hs = altezza(p.left)
	hd = altezza(p.right)
	return max(hs, hd) + 1

def inserisci(p, A, x=0):
	A[x] = p.key
	if p.left: inserisci(p.left, 2*x+1)
	if p.right: inserisci(p.right, 2*x+2)
```

---
# ES.15
## 1.
Dato un puntatore alla radice di un albero binario di ricerca `p` e un elemento `x` in esso contenuto trovare il successivo, ovvero l’elemento più piccolo tra quelli maggiori di `x`
```python
def es(p, x):
	while p!=None:
		if x<p.key: p = p.left
		elif x>p.key: p = p.right
		else: p=p.right; break
	
	while p!=None:
		if p.left:
			p = p.left
		else:
			return p.key
```

## 2.
Dato un puntatore alla radice di un albero binario di ricerca `p` e un elemento `x` in esso contenuto trovare il predecessore, ovvero l’elemento più grande tra quelli minori di `x`
```python
def es(p, x):
	while p!=None:
		if x<p.key: p = p.left
		elif x>p.key: p = p.right
		else: p=p.left; break
	
	while p!=None:
		if p.right: p = p.right
		else: return p.key
```