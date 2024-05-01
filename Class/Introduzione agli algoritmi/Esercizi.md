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
	while (P != None):
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
