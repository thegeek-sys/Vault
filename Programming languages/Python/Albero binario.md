---
Created: 2023-12-11
Programming language: "[[Python]]"
Related:
  - "[[Ricorsione]]"
---
---
## Introdution
![[Binary_tree.jpg]]
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
	Hsx = 0 if not self.sx else self.sx.height()
	Hdx = 0 if not self.dx else self.dx.height()
	D_root = Hsx + Hdx + 1
	# calcolo il percorso max su rami destri
```