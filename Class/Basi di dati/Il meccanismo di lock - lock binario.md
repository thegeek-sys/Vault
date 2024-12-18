---
Created: 2024-12-18
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Introduction
Per **lock** si intende il **privilegio di accesso** ad un **singolo item** realizzato mediante una variabile associata all’item (variabile lucchetto) il cui valore descrive lo stato dell’item rispetto alle operazioni che possono essere effettuate su di esso (ogni item può essere locked o unlocked)

Nella sua forma più semplice, un lock:
- viene **richiesto** da una transazione mediante un’operazione di *locking* tramite la quale se il valore della variabile è unlocked la transazione può accedere all’item e alla variabile viene assegnato il valore locked
- viene **rilasciato** da una transazione mediante un’operazione di *unlocking* che assegna alla variabile il valore unlocked

Quindi il locking agisce come **primitiva di sincronizzazione**, cioè se una transazione richiede un lock su un item su cui un’altra transazione mantiene un lock, la transazione non può procedere finché il lock non viene rilasciato dalla prima transazione
Fra l’esecuzioen di un’operazione di locking su un certo item $X$ e l’esecuzione di un’operazione di unlocking su $X$ la transazione **mantiene un lock su $X$**

---
## Schedule legale
Uno schedule è detto **legale** se una transazione effettua un **locking ogni volta che deve scrivere/leggere** un item e se ciascuna transazione **rilascia ogni lock** che ha ottenuto

---
## Lock binario
Un lock **binario** può assumere solo due valori: **locked** e **unlocked**

Le transazioni fanno uso di due operazioni:
- $lock(X)$ → per richiedere l’accesso all’item $X$
- $unlock(X)$ → per rilasciare l’item $X$ consentendone l’accesso ad altre transazioni

L’insieme degli item letti e quello degli item scritti da una transazione coincidono

### Esempio
Risolviamo il primo dei problemi visti, cioè l’update loss
![[Screenshot 2024-12-18 alle 21.22.35.png|300]]
![[Pasted image 20241218212457.png|200]]

Riscriviamo le transazioni utilizzando le primitive del lock binario
![[Pasted image 20241218223002.png|300]]

Vediamo ora uno schedule legale di $T_{1}$ e $T_{2}$ che risolve il problema dell’update loss
![[Pasted image 20241218223108.png|250]]

---
## Modello per le transazioni
![[Pasted image 20241218223305.png|center|180]]

Una transazione è una **sequenza di operazioni di lock e unlock**:
- ogni $lock(X)$ implica la **lettura** di $X$
- ogni $unlock(X)$ implica la **scrittura** di $X$

![[Pasted image 20241218223446.png|center|220]]

In corrispondenza di una scrittura viene associato un nuovo valore coinvolto che viene calcolato da una **funzione** che è associata in modo **univoco** ad ogni coppia lock-unlock e che ha come **argomenti tutti gli item letti** (locked) **dalla transazione prima dell’operazione di unlock** (perché magari i loro valori hanno contribuito all’aggiornamento dell’item corrente)

---
## Equivalenza
Due schedule sono **equivalenti** se le formule che danno i valori finali per ciascun item sono le stesse 

>[!warning] Le formule devono essere uguali per tutti gli item

>[!info]
>Vedremo che la proprietà di equivalenza degli schedule dipende dal protocollo di locking usato
>Adottiamo un **modello delle transazioni** per poter astrarre delle specifiche operazioni che si basa su quelle rilevanti per valutare le sequenze degli accessi, cioè in questo caso lock e unlock

---
## Schedule serializzabile
Uno schedule è **serializzabile** se è equivalente ad uno schedule seriale (basta trovarne uno)

### Esempio
Consideriamo le due transazioni
![[Pasted image 20241218224919.png|500]]

e lo schedule
![[Pasted image 20241218225014.png|550]]

Considerando $X_{0}$ il valore iniziale di $X$ e $Y_{0}$ il valore iniziale di $Y$ allora $f_{4}(f_{1}(X_{0}),Y_{0})$ è il valore finale di $X$
