---
Created: 2024-03-07
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduzione
Esistono due tipi di memoria:
- **heap**: Vengono allocate le aree di memoria per la _creazione dinamica_ (oggetti)
- **stack**: Vengono allocate le variabili locali

![[Screenshot 2024-03-07 alle 13.35.24.png]]![[Screenshot 2024-03-07 alle 13.35.33.png]] ![[Screenshot 2024-03-07 alle 13.35.41.png]]

---
## Static
I campi di una classe possono essere dichiarati static (relativo dell’intera classe).
Un campo static esiste **una sola locazione di memoria**, allocata prima di qualsiasi oggetto della classe in una zona speciale di memoria nativa chiamata *MetaSpace*. Viceversa, per ogni campo non static esiste una locazione di memoria per ogni oggetto, allocata a seguito dell’istruzione new

---
## Istruzioni all’avvio del compilatore
Quando viene avviato un programma il compilatore di Java esegue queste azioni:
- Tramite il **ClassLoader** carica la classe e vede se c'è un metodo main eseguibile.
- Alloca nel **MetaSpace** i campi statici se ci sono
- Inizia ad eseguire il main riservandogli una locazione di memoria nello **stack** e copiando **nell'heap** i valori presenti nella lista args, anche se vuota.
- Va avanti nella compilazione del programma allocando locazioni di memoria di variabili, oggetti e metodi mano a mano che li troviamo.

I valori dei parametri di un metodo vengono sempre copiato e non passati per riferimento, se troviamo un nuovo metodo questo viene allocato ad un livello superiore nello stack.