---
Created: 2024-03-12
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
In Java possiamo prendere decisioni attraverso istruzioni di controllo **condizionali** (istruzioni che possono essere o non eseguite sulla base di certe condizioni) e **iterative** (istruzioni che devono essere eseguite ripetutamente sulla base di certe condizioni)

---
## if
Per realizzare una decisione si usa l’istruzione `if`. La sintassi è:
```java
if (<espressione booleana>) <singola istruzione>;
```

Oppure:
```java
if (<espressione booleana>) {
	<istruzioni caso true>;
} else {
	<istruzioni caso false>;
}
```

E’ inoltre importante ricordare che l’else, quando non vengono utilizzate le graffe, si riferisce sempre all’istruzione if immediatamente precedente