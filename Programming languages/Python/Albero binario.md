---
Created: 2023-12-11
Programming language: "[[Python]]"
Related:
  - "[[Ricorsione]]"
---
---
## Introdution
![[Binary_tree.jpg]]
E’ possibile anche utilizzare delle classi per memorizzare la stuttura dati ad albero di una determinata funzione ricorsiva. Un albero è una moltitudine di oggetti di nodi binari.
Un albero binario è composto da:
- radice (root) → l’origine dell’albero
- nodi
- foglie (leaves)

```python
class BinaryNode:
	
	def __init__(self, value, sx=None, dx=None):
		self.value = value
		self.sx = sx
		self.dx = dx

root = BinaryNode(1, BinaryNode(6, BinaryNode(2), BinaryNode(3)),
					 BinaryNode(7, BinaryNode(4), BinaryNode(5)))
```

---
## Altezza
Per calcolarla devo pensarlo ricorsivamente ed ritornare sul ritorno l’altezza massima rispettiva all’albero di sinistra e di destra e sommare 1

```python
def height(self):
	Hsx = 0 if not self.sx else self.sx.height()
	Hdx = 0 if not self.dx else self.dx.height()
	return max(Hsx, Hdx) + 1
```

---
## Diametro (percorso massimo)

```python
def diameter(self):
	# calcolo il percorso massimo sulla radice
	Hsx = 0 if not self.sx else self.sx.height()
	Hdx = 0 if not self.dx else self.dx.height()
	D_root = Hsx + Hdx + 1
	# calcolo il percorso max sul ramo destro
	D_dx = 0 if not self.dx else self.dx.diameter()
	# poi sul sinistro
	D_sx = 0 if not self.sx else self.sx.diameter()
	# prendo il massimo tra i tre dato che il più lungo potrebbe non passare
	# per la radice
	return max(D_root, D_dx, D_sx)
```

## Ricerca
La ricerca all’interno di un albero può essere eseguita in modo ricorsivo. Il caso base lo ho se mi trovo nella root dell’albero e ho il valore ricercato, altrimenti dovrò entrare ricorsivamente in ogni ramo dell’albero

```python
def find(self, value):
	if self.value == value:
		return True
	# non esistono nè albero sinistro nè destro
	elif not self.sx and not self.dx:
		return False
	# esiste sottoalbero sinistro
	elif self.sx:
		if self.sx.find(value):
			return True
	# esiste sottoalbero destro
	elif self.dx:
		return self.dx.find(value)
	else:
		return False
```

---
## Ricerca DFS
La ricerca DFS (Depth First Search) può essere di due tipi:
- in Pre-Order
- in Post-Order
### Pre-Order
Nella ricerca in pre-order segue questo ordine per la ricerca del valore:
1. Nodo
2. Sotto-albero sx
3. Sotto-albero dx
In particolare risponde in ordine alle seguenti domande:
- È nel valore del nodo corrente? Se sì ho fatto e trovato
- Altrimenti, ho dei figli da controllare? Se non ho figli, ho finito e non ho trovato

```python
def find(self, value):
	if self.value == value:
		return True
	elif not self.sx and not self.dx:
		return False
	elif self.sx:
		if self.sx.find(value):
			return True

```