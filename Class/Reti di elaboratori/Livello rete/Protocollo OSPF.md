---
Created: 2025-04-30
Class: "[[Reti di elaboratori]]"
Related:
  - "[[Livello rete]]"
---
---
## Link state
Lo stato di un link indica il costo associato al link. Se il costo è $\infty$ significa che il collegamento non esiste oppure è stato interrotto

Ogni nodo deve conoscere i costi di tutti i collegamenti della rete (nel distance vector invece si usavano soli i vicini). Il link state database mantiene la mappa completa della rete

---
## Link state database (LSDB)
Il link-state database è unico per tutta la rete e ogni nodo ne possiede una copia

![[Pasted image 20250430215936.png|300]]

Viene rappresentato tramite una matrice
![[Pasted image 20250430220006.png]]

### Come può un nodo costruire il LSDB?