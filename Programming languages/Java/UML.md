---
Created: 2024-03-05
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Introduzione
UML (*Unified Modeling Language*) serve per implementare in maniera standard dei diagrammi. In particolare può essere utilizzato per rappresentare le classi e le interazioni tra di esse.

---
## Diagrammi delle classi
![[Screenshot 2024-03-05 alle 17.35.59.png|450]]

- **`+`** → visibilità pubblica
- **`-`** → visibilità privata
- **`#`** → visibilità protetta
- **`~`** → visibilità package

---
## Dipendenza tra classi
Posso utilizzare la freccia piena in modo tale da indicare la **dipendenza generica** tra due classi.

![[Screenshot 2024-03-13 alle 09.08.35.png|center|400]]

---
## Classe astratta
In UML una classe astratta la si indica scrivendo il nome in corsivi e/o si usa il tag \<abstract\>

![[Screenshot 2024-03-25 alle 22.06.56.png|center|250]]

---
## Ereditarietà tra due classi
Per indicare che una classe è sottoclasse di una superclasse in UML devo utilizzare questo tipo particolare di freccia. Quando in una sottoclasse definisco nuovamente un metodo della superclasse vuol dire che la sto ridefinendo conferendogli delle caratteristiche più specifiche

![[Screenshot 2024-03-19 alle 10.32.45.png|center|250]]
