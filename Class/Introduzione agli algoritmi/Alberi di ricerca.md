---
Created: 2024-05-07
Class: Introduzione agli algoritmi
Related: 
Completed:
---
---
## Introduction
![[Screenshot 2024-05-07 alle 16.29.47.png|center|250]]
Un **albero binario di ricerca bilanciato** (*red-black tree*) è un albero nel quale vengono mantenute le seguenti proprietà:
- ogni nodo contiene una chiave
- il valore della chiave contenuta in ogni nodo è maggiore della chiave contenuta in ciascun nodo del suo sottoalbero sinistro (se esiste)
- il valore della chiave contenuta in ogni nodo è minore della chiave contenuta in ciascun nodo del suo sottoalbero destro (se esiste)

Come gli alberi già visti in precedenza anche questi supportano le operazioni di:
- `ricerca` → $O(\log(n))$
- `inserimento` → $O(\log(n))$
- `cancellazione` → $O(\log(n))$
Queste operazioni, poiché sono effettuate su alberi bilanciati, impiegano esattamente $O(h)$

>[!hint]
> Poiché sto operando su alberi binari di ricerca **bilanciati** per le operazioni che comportano il modificare la struttura, devo ricordarmi di **ribilnaciare** l’albero


![[Screenshot 2024-05-07 alle 16.48.47.png|center|130]]
A differenza dagli alberi binari classici, questi oltre ad avere i riferimenti ai due rami figli, introducono un ulteriore campo rappresentato da `parent`, ovvero un riferimento al puntatore padre (ci tornerà molto utile nell’operazione di cancellazione)

### Equazione di ricorrenza
$$
\begin{cases}
T(h)\leq T(h-1)+\theta(1) \\
T(0)=\theta(1)
\end{cases}
$$
Questo sistema rappresenta l’equazione di ricorrenza per le operazioni effettuate su un albero di ricerca. Nel **worst case** avremo che $T(h)=T(h-1)+\theta(1)$

---
## Stampare albero
Stampando un albero di ricerca **in-order** avremo tutte le chiavi in ordine crescente

![[Screenshot 2024-05-07 alle 16.41.22.png|200]]
```python
def stampaABR(p):
	if p != None:
		stampaABR(p.left)
		print(p.key)
		stampaABR(p.right)

# 6 9 12 33 38 39 40 42 45 48 50 52 54 57 60 68
```

---
## Massimo e minimo
Dalla struttura stessa dell’albero di ricerca è facile capire che il massimo si trova nel nodo più a destra dell’albero

```python
def massimo(p):
	q = p
	while q.right:
		q = q.right
	return q.key
```

E che, al contrario, il minimo si trova nel nodo più a sinistra
```python
def minimo(p):
	q = p
	while q.left:
		q = q.left
	return q.key
```

---
## Ricerca
Quest’operazione è sostanzialmente identica a quella eseguita sugli alberi binari

```python
def ricerca(p, x):
	if p==None: return False
	if p.key==x: return True
	if x <= p.key: return ricerca(p.left, x)
	else: return ricerca(p.right, x)
```

---
## Inserimento
L’inserimento procede come segue:
- si esegue une discesa che viene guidata dai valori memorizzati nei nodi che si incontrano lungo il cammino
- quando si arriva al punto di voler proseguire la discesa verso un puntatore vuoto allora, in quella posizione, si aggiunge un nuovo nodo contenente il valore da inserire.

Nella foto seguente viene mostrato il come viene esplorato l’albero nel caso in cui si volesse aggiungere $72$ ad un albero già esistente
![[3C99BF95-6C66-4706-9DF9-C39099D45531.jpeg]]

```python
def inserimento(p, x):
	z = NodoABR(x)
	if p==None: return z
	q = p
	while True:
		if x>p.key and p.right:
			p = p.right
		elif x<p.key and p.left:
			p = p.left
		else break
	if x<p.key:
		p.left = z
	elif x>p.key:
		p.right = z
	return q
```

---
## Cancellazione
