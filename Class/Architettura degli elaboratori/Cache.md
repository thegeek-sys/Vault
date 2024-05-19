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

![[Screenshot 2024-05-17 alle 13.08.32.png|470]]

La **gerarchia delle memoria** consiste in un insieme di livelli di memoria, ciascuno caratterizzato da una **diversa velocità e dimensione**: a parità di capacità, le memorie più veloci hanno un costo più elevato per singolo bit di quelle più lente, perciò esse sono di solito più piccole

![[Screenshot 2024-05-17 alle 13.41.35.png]]

> [!info]
> La memoria più veloce è posta infatti più vicino al processore di quella lenta in modo tale fa fornire all’utente una quantità di memoria pari a quella disponibile nella tecnologia più economica, consentendo allo stesso tempo una velocità di accesso pari a quella garantita dalla memoria più veloce

Anche i **dati** sono organizzati in modo gerarchico: un livello più vicino al processore contiene in generale un sottoinsieme dei dati memorizzati in ognuno dei livelli sottostanti, e tutti i dati si trovano memorizzati nel livello più basso della memoria (es. i libri sulla scrivania dello studente sono un sottoinsieme dei libri della biblioteca che è un sottoinsieme delle biblioteche universitarie, più ci si allontana dal processore più aumenta il tempo per leggere i dati proprio come per lo studente)

![[Screenshot 2024-05-17 alle 13.25.42.png|250]]

Una gerarchia delle memorie può essere composta da più livelli, ma i dati vengono di volta in volta trasferiti solo tra due livelli vicini.
La più piccola quantità di informazione che può essere presente o assente in questa gerarchia su due livelli è detta **blocco** o **linea** (es. i libri per lo studente)

Nel caso di successo o insuccesso nell’accedere al dato richiesto abbiamo diverse terminologie:
- *hit* → il dato si trova in uno dei blocchi presenti nel livello superiore (es. studente trova informazioni in uno dei libri che ha sulla scrivania)
- *miss* → il dato non si trova nel livello superiore della gerarchia, in questo caso il dato va ricercato nel livello inferiore (es. studente si alza dalla scrivania e va a cercare sugli scaffali)
- *hit rate* → numero di hit fratto numero totale di accessi in memoria
- *miss rate* → numero di miss fratto il numero totale di accessi in memoria
- *tempo di hit* → tempo per tentare di trovare il dato nel livello superiore, che esso abbia successo o che non non sia presente (es. tempo che impiega lo studente a passare in rassegna i libri)
- *penalità di miss* → tempo necessario a sostituire un blocco del livello superiore con un nuovo blocco caricato dal livello inferiore della gerarchia e a trasferire i dati contenuti in questo blocco al processore (es. tempo che serve per prendere un nuovo libro dagli scaffali e metterlo sulla scrivania). Generalmente il tempo di hit è molto minore della penalità di miss

---
## Cache Direct-Mapped
Poiché una cache deve contenere solo i dati più richiesti utilizzando dimensioni limi- tate, è necessario che più blocchi di memoria vengano **salvati nello stesso spazio**, sovrascrivendosi a vicenda.

![[Screenshot 2024-05-18 alle 21.30.11.png|center|400]]

Strutturiamo quindi la nostra cache come composta da un numero N di linee, corrispondenti agli spazi occupabili dai blocchi, dove ogni linea è composta da:
- **bit di validità** → indicante se i dati contenuti nella linea siano validi o meno. Se tale bit vale 0, allora la linea viene considerata come “vuota”
- **campo tag** → in grado di distinguere quale blocco della memoria sia caricato nella linea. Tale campo risulta fondamentale poiché più blocchi in memoria vengono mappati sulla stessa linea, prevenendo la lettura del blocco sbagliato
- **blocco** → stesso memorizzato all’interno della linea

![[Screenshot 2024-05-19 alle 10.46.19.png|600]]

A questo punto, ci serve un modo matematico per poter calcolare i singoli valori che ci permettono di lavorare sulle linee della cache a partire dall’indirizzo di memoria richiesto.

Prima, però, è necessario puntualizzare che:
- Per praticità, realizziamo la nostra cache con $\mathbf{2^n}$ **linee**, associando ad ognuna di esse un indice. Per selezionare uno di tali indici, quindi, sono necessari $n$ bit.
- Scomponiamo la memoria in blocchi da $\mathbf{2^m}$ **word**, dove ogni word corrisponde a 4 byte. La dimensione di ogni blocco, quindi, risulta essere $2^m \cdot 4\cdot8$ bit.
- Abbiamo bisogno di un valore, chiamato **offset di word**, che possa indicare quale word interna al blocco corrisponda a quella richiesta dall’indirizzo di memoria. Tale valore, quindi, avrà una dimensione di $m$ bit, che ci permettono di selezionare una delle $2^m$ word.
- Analogamente, abbiamo bisogno di un valore, chiamato **offset di byte** che vada a selezionare quale dei 4 byte componenti tale word corrisponda al byte specifico richiesto dall’indirizzo di memoria. Poiché ogni word è sempre composta da 4 byte, saranno necessari **2 bit** per tale valore.

![[Screenshot 2024-05-19 alle 10.57.06.png]]
Facendo i conti la **dimensione totale della cache** risulta essere $17088 \text{ bit}=2136\text{ byte}\approx 2.1 \text{ KB}$

### Determinare un HIT o un MISS
Una volta identificata la struttura dei campi, possiamo utilizzarli per realizzare la vera e propria cache, il cui funzionamento può essere riassunto nella seguente schematica:
![[Screenshot 2024-05-19 alle 11.05.08.png]]

Oltre all’uso della circuiteria, possiamo calcolare matematicamente se venga effettuato un HIT o un MISS utilizzando le dimensioni dei vari campi individuati

Per calcolare il **numero di blocco**, è necessario **shiftare a destra l’indirizzo di memoria** di una quantità di bit pari alla **dimensione dell’offset di blocco**, ossia $m+2$, in modo da poterli "scartare", considerando così solo i $32−m−2$ bit riservati al numero di blocco:

$$
\verb|#Blocco| = \verb|Address|>>m+2
$$
Tuttavia, ricordiamo che uno shift a destra di $x$ posizioni equivale a dividere il valore stesso per $2^x$ (arrotondamento per difetto):
$$
\verb|#Blocco| = \verb|Address|>>m+2 =\left\lfloor \frac{\verb|Address|}{2^{m+2}} \right\rfloor
$$
Di fatti notiamo come $2^{m+2}$ corrisponda esattamente al **numero di byte del blocco** ($2^{m+2}=2^m\cdot_{4}$, dove $2^m$ ricordiamo essere il numero di word del blocco). Dunque il numero di blocco dell’indirizzamento di memoria richiesto dall’accesso corrisponde a:
$$
\verb|#Blocco| = \left\lfloor \frac{\verb|Address|}{2^{m+2}} \right\rfloor = \left\lfloor \frac{\verb|Address|}{\verb|Num. byte blocco|} \right\rfloor
$$
