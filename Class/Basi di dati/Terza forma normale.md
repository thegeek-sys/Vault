---
Created: 2024-10-27
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Quali sono i problemi di uno schema mal progettato?
Ritorniamo al nostro esempio di base di dati che contiene le informazioni sugli studenti e sugli esami sostenuti, e ripartiamo dalla soluzione “buona” tornata alla fine

![[Progettazione di una base di dati relazionale - Problemi e vincoli#Ipotesi 3]]

### Considerazioni
Poiché il numero di matricola identifica univocamente uno studente, ad ogni numero di matricola corrisponde:
- un solo codice fiscale ($\text{Matr}\to \text{CF}$)
- un solo cognome ($\text{Matr}\to \text{Cogn}$)
- un solo nome ($\text{Matr}\to \text{Nome}$)
- una sola data di nascita ($\text{Matr}\to \text{Data}$)
- un solo comune di nascita ($\text{Matr} \to \text{Com}$)

Quindi un’istanza di $\text{Studente}$ per essere legale deve soddisfare la dipendenza funzionale
$$
\text{Matr}\to \text{Matr, CF, Cogn, Nome, Data, Com}
$$

Con considerazioni analoghe abbiamo che un’istanza di $\text{Studente}$ per essere legale deve soddisfare la dipendenza funzionale
$$
\text{CF}\to \text{Matr, CF, Cogn, Nome, Data, Com}
$$
Pertanto sia $\text{Matr}$ che $\text{CF}$ sono chiavi di $\text{Studente}$

D’altra parte possiamo osservare che ci possono essere due studenti con lo stesso cognome e nomi differenti quindi possiamo avere due istanze di $\text{Studente}$ che **non** soddisfano le dipendenza funzionale $\text{Cogn}\to \text{Nome}$. Inoltre possiamo avere istanze di $\text{Studente}$ che non soddisfano: $\text{Cogn}\to \text{Nome}$, $\text{Cogn}\to \text{Data}$, $\text{Cogn}\to \text{Com}$ ecc.

Con ciò possiamo concludere che le **uniche dipendenze funzionali non banali** che devono essere soddisfatte da un’istanza legale di $\text{Studente}$ sono del tipo
$$
K\to X
$$
dove $K$ contiene una chiave ($\text{Matr}$ o $\text{CF}$)

>[!hint]
>Vedremo che questa è la prima condizione che però va ulteriormente rifinita per arrivare ad una definizione precisa di terza forma normale (3NF)
