---
Created: 2024-05-07
Class: Introduzione agli algoritmi
Related: 
Completed:
---
---
## Introduction
![[Screenshot 2024-05-07 alle 16.29.47.png|center|250]]
Un **albero binario di ricerca** è un albero nel quale vengono mantenute le seguenti proprietà:
- ogni nodo contiene una chiave
- il valore della chiave contenuta in ogni nodo è maggiore della chiave contenuta in ciascun nodo del suo sottoalbero sinistro (se esiste)
- il valore della chiave contenuta in ogni nodo è minore della chiave contenuta in ciascun nodo del suo sottoalbero destro (se esiste)

Come gli alberi già visti in precedenza anche questi supportano le operazioni di:
- `ricerca` → $O(\log(n))$
- `inserimento` → $O(\log(n))$
- `cancellazione` → $O(\log(n))$
Queste operazioni, nel caso in cui vengano effettuate su alberi bilanciati impiegano esattamente $O(h)$

>[!hint]
> Poiché sto operando su alberi binari di ricerca **bilanciati** per le operazioni che comportano il modificare la struttura, devo ricordarmi di **ribilnaciare** l’albero

---
## Stampare albero
Stampando un albero di ricerca **in-order** avremo tutte le chiavi in ordine crescente

```python
def stampaABR(p):
	if p != None:
		stampaABR(p.left)
		print(p.key)
		stampaABR(p.right)

# 
```