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