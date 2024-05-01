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