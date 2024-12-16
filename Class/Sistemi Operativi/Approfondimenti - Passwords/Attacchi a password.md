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

L’attacco del dizionario consiste nel compilare una lista di password comunemente utilizzata e fare un *bruteforce*