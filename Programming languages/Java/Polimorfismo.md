---
Created: 2024-03-26
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduction
Il polimorfismo è uno dei punti cardine della programmazione orientata agli oggetti oltre all’ereditarietà. Questo dunque ci permette di utilizzare un metodo senza dover conosce il tipo esatto (la classe) su cui si invoca il metodo

Una variabile di un certo tipo a può contenere un riferimento a un oggetto del tipo A o di qualsiasi sua sottoclasse
```java
// a è il riferimento di animale
// ma ho usato il costruttore di Gatto
Animale a = new Gatto();
a = new Chihuahua();
```

La selezione del metodo da chiamare avviene in base all’effettivo tipo dell’oggetto riferito alla variabile
```java
Animale a = new Gatto();
a.emettiVerso(); // "miaoo"
a = new Chihuahua();
a.emettiVerso(); // "bau bau"
```

---
## Binding
Per **binding** in programmazione si intende associare ad ogni variabile il proprio tipo
### Statico
Il **binding statico** consiste nell’associare una variabile al proprio tipo, e viene svolto in java dal compilatore che creerà quindi una tabella dei binding statici
Questo viene fatto, in Java così negli altri linguaggi compilati senza eseguire il codice ma solo “osservandolo”

### Dinamico
Il polimorfisrmo, come implementato in java, vede la JVM elaborare il **binding dinamico**, poiché l’associazione tra una variabile di riferimento e un metodo da chiamare viene stabilita a tempo di esecuzione.
Questo viene solitamente utilizzato dai linguaggi interpretati (come Python) e in Java viene utilizzato quando, attraverso il polimorfismo, utilizzo il costruttore di una sottoclasse del tipo di definizione oppure quando chiamo dei metodi
