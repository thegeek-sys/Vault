---
Created: 2024-05-17
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## Introduction
Per capire al meglio la Cache possiamo prendere un esempio concreto. Immaginiamo uno studente che si sta preparando ad un esame e si trova in biblioteca.
Lo studente per avere tutte le fonti da cui studiare la materia, decide di fare una prima scrematura raccogliendo sulla propria scrivania tutti i libri della biblioteca che sono concerni a ciò che sta studiando.
Allo stesso modo il calcolatore per rendere più veloce l’accesso alla memoria decide di pre-caricarsi le informazioni che sa gli serviranno durante l’esecuzione, proprio come lo studente ha raccolto la piccola quantità di libri necessaria per lo studio in modo tale da non dover perdere tempo ogni volta che cambia argomento a riporre il libro precedente e cercare un’ulteriore libro dentro la biblioteca.
Questo miglioramento però non è presente nel caso in cui lo studente decidesse di mettere sulla propria scrivania tutti i libri presenti in biblioteca perché si ritroverebbe costretto a consultare tutti i libri per trovare le informazioni che gli interessano; allo stesso modo anche l’elaboratore decide di non caricarsi tutto ciò che è presente in memoria, ma solamente la parte che gli interessa

---
## Principio della località
Il **principio di località** sta alla base del comportamento dei programmi in un calcolatore ed è del tutto simile al modo di cercare informazioni in una biblioteca. Questo principio afferma che un programma, in un certo istante di tempo, accede soltanto ad una porzione relativamente piccola del sui spazio di indirizzamento, proprio come lo studente accede solo a una piccola porzione dei libri della biblioteca.

Esistono due tipi di località:
- **località temporale** (località nel tempo) → quando si fa riferimento ad  un elemento, c’è la tendenza a fare riferimento allo stesso elemento dopo poco tempo (quando lo studente prende il libro dalla libreria si per consultarlo si suppone che dopo poco lo dovrà fare nuovamente)
- **località spaziale** (località nello spazio) → quando si fa riferimento a un elemento, c’è la tendenza a fare riferimento poco dopo ad altri elementi che hanno l’indirizzo vicino ad esso (generalmente la biblioteca è organizzata in modo tale da avere i libri con stessa tipologia di argomento nella stessa sezione in modo tale che lo studente, dopo aver preso il libro che cerca, è facilitato a cercare libri con argomenti simili)

La località emerge in modo naturale nelle **strutture di controllo semplici** e tipiche dei programmi

>[!hint] Esempi
>In un `for` le istruzioni e i dati che si trovano al suo interno vengono letti ripetutamente dalla memoria → alta località temporale
>
>Dato che le istruzioni di un programma generalmente vengono caricate in sequenza dalla memoria, i programmi presentano un’alta località spaziale

---
## Gerarchia delle memorie
Si usufruisce del principio di località sfruttando la memoria di un calcolatore in forma gerarchica.
La **gerarchia delle memoria** consiste in un insieme di livelli di memoria, ciascuno caratterizzato da una **diversa velocità e dimensione**: a parità di capacità