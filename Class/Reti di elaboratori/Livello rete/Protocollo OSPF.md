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
Ogni nodo della rete deve innanzitutto conoscere i propri vicini e i costi dei collegamenti verso di loro. Per cui ogni nodo invia un messaggio di `hello` a tutti i suoi vicini di conseguenza ogni nodo riceve gli `hello` dei vicini e crea la lista dei vicini con i relativi costi dei collegamenti

La lista (vicino, costo) viene chiamata **LS packet** (*LSP*)
Si può dire dunque che ogni nodo esegue un *flooding* degli LSP:
- invia a tutti i vicini il proprio LSP
- quando riceve l’LSP di un vicino, se è un nuovo LSP allora lo inoltra a tutti i suoi vicini eccetto quello da cui lo ha ricevuto

