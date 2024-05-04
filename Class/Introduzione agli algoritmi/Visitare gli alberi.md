---
Created: 2024-05-03
Class: Introduzione agli algoritmi
Related:
  - "[[Class/Introduzione agli algoritmi/Strutture dati#Alberi|Alberi]]"
---
---
## Introduction
Un’operazione basilare sugli alberi è l’accesso a tutti i suoi nodi, uno dopo l’altro, al fine di poter effettuare una specifica operazione su ciascun nodo.
Tale operazione sulle liste si effettua con una semplice iterazione, ma sugli alberi la situazione è più complessa dato che la loro struttura è ben più articolata.
L’accesso progressivo a tutti i nodi di un albero si chiama **visita dell’albero**.

![[2D6B7EFC-C27C-4E21-9296-3DB8C45A8F92.jpeg|center|200]]
Negli alberi binari si ha la possibilità di visitare l’albero in tre differenti maniere:
- **in pre-order** → il nodo è visitato prima di proseguire la visita nei suoi sottoalberi
- **in order** → il nodo è visitato dopo la visita del sottoalbero sinistro e prima di quella del sottoalbero destro
- **in post-order** → il nodo è visitato dopo entrambe le visite dei sottoalberi

### Pre-order
```python
def stampaAlbero(p):
	if p==None return None
	print(p.key)
	stampaAlbero(p.left)
	stampaAlbero(p.right)

# -> 3, 2, 7, 1, 5
```

### Order
```python
def stampaAlbero(p):
	if p==None return None
	stampaAlbero(p.left)
	print(p.key)
	stampaAlbero(p.right)

# -> 2, 1, 7, 5, 3
```

### Post-order
```python
def stampaAlbero(p):
	if p==None return None
	stampaAlbero(p.left)
	stampaAlbero(p.right)
	print(p.key)

# -> 1, 5, 7, 2, 3-
```

---
## Costo computazionale delle visite
Il costo computazionale delle visite dell’albero è uguale per tutti e tre i tipi di visita, ma varia al variare della struttura dati utilizzata per memorizzare l’albero.

Nel caso di memorizzazione tramite record e puntatori si ha
$$
\begin{gather}
T(n)=T(k)+T(n-1-k)+\theta(1) \\
T(0)=\theta(1)
\end{gather}
$$

Quando l’albero è **completo** si ha che $k\approx\frac{n}{2}$ e l’equazione diviene:
$$
T(n) = 2T \left(\frac{n}{2}\right)+\theta(1)
$$
Che ricade nel caso 1 del Master theorem e quindi ha soluzione:
$$T(n)=\theta(n^{\log_{2}2})=\theta(n)$$


Quando l’albero è massivamente **sbilanciato** si verifica che $k=0$ (oppure $n-1-k=0$), per cui si ottiene l’equazione:
$$
T(n)=T(n-1)+\theta(1)
$$
Risolvibile banalmente tramite metodo iterativo:
$$
T(n)=n\cdot \theta(1)=\theta(n)
$$

Dai due esempi sopra proposti è facile capire che indipendentemente dalla forma dell’albero la complessità risulta essere sempre $\theta(n)$. Dimostriamolo tramite il metodo di sostituzione.
Eliminiamo quindi la notazione asintotica:
$$\begin{cases}
T(n)=T(k)+T(n-1-k)+b \\
T(1)=a
\end{cases}$$
Testiamo quindi le soluzioni:
$$
\begin{gather}
T(n)\leq c \cdot n\\
T(n)\geq c\cdot n
\end{gather}
$$
Passo base: $T(1)=a\leq c$ che è vero se prendiamo $c\geq a$
Passo induttivo:
$$
\begin{align}
T(n)\leq ck+c(n-1-k)+b &= \\
&=c(n-1)+b \\
&=c\cdot n-c+b\\
&\leq c\cdot n
\end{align}
$$
