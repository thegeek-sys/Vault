---
Created: 2024-03-07
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Memoria

[](https://github.com/alem1105/Quartz/blob/v4/content/Secondo%20Semestre/Metodologie%20di%20Programmazione/Metodologie%20di%20Programmazione%20-%20Lezione%203.md#memoria)

Esistono due tipi di memoria:

- **heap**: Vengono allocate le aree di memoria per la _creazione dinamica_ (oggetti)
- **stack**: Vengono allocate le variabili locali

![[Screenshot 2024-03-07 alle 13.35.24.png]] ![[Screenshot 2024-03-07 alle 13.35.33.png]] ![[Screenshot 2024-03-07 alle 13.35.41.png]]

- **MetaSpace**: Qui vengono allocati i campi di tipo static relativi all'intera classe, nel caso di un oggetto non static invece abbiamo una locazione di memoria diversa per ogni oggetto creato.

Quando viene avviato un programma il compilatore di Java esegue queste azioni:

- Tramite il **ClassLoader** carica la classe e vede se c'è un metodo main eseguibile.
- Alloca nel **MetaSpace** i campi statici se ci sono
- Inizia ad eseguire il main riservandogli una locazione di memoria nello **stack** e copiando **nell'heap** i valori presenti nella lista args, anche se vuota.
- Va avanti nella compilazione del programma allocando locazioni di memoria di variabili, oggetti e metodi mano a mano che li troviamo.

I valori dei parametri di un metodo vengono sempre copiato e non passati per riferimento, se troviamo un nuovo metodo questo viene allocato ad un livello superiore nello stack.