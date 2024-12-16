---
Created: 2024-12-16
Class: "[[Sistemi Operativi]]"
Related:
  - "[[Approfondimenti - Passwords]]"
Completed:
---
---
## Introduction
Nonostante le funzioni di hash siano irreversibili, risultano comunque attaccabili.
Sono due i principali attacchi alle funzioni hash: **attacco dizionario**, **attacco rainbow table**

---
## Attacco dizionario
Uno dei problemi è che ci sono un numero infinito di password che possono risultare nello stesso hash. Come facciamo quindi a scoprire qual è quella giusta?

Fortunatamente per gli attaccanti gli utenti non vogliono (o non possono) ricordarsi password complesse e non vogliono ricordarsene molte diverse
Come conseguenze si ha che le password risultano brevi e semplici e che la stessa password probabilmente è riusata molte volte per servizi diversi

L’attacco del dizionario consiste nel compilare una lista di password comunemente utilizzata e fare un *bruteforce*. Dunque per ogni password nella mia lista, calcolo l’hash e verifico se è uguale a quello della password

>[!info]
>Esistono oggi dizionari di svariati GB di password (es. rockyou)

**Vantaggi**
- Molto semplice da effettuare (richiede solo una lista di password)
- Versatile → funziona per qualsiasi funzione hash, password, ecc
- Moltissimi tool disponibili per automatizzare il tutto (es. John the Ripper)
**Svantaggi**
- Può essere molto lento, in quanto richiede la computazione in real time degli hash
- La password può non essere presente nel dizionario

---
## Attacco Rainbow Table
