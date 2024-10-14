---
Created: 2024-10-14
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Obbiettivo
Supponiamo di voler creare una base di dati contenente i seguenti dati di studenti universitari:

Dati anagrafici e indicativi
- nome e cognome
- data, comune e provincia di nascita
- matricola
- codice fiscale

Dati curriculari
- per ogni esame sostenuto
	- voto
	- data
	- codice
	- titolo
	- docente del corso

## Ipotesi 1
La base di dati consiste di una sola relazione con schema
$$
\text{Curriculum(Matr, CF, Cogn, Nome, DataN, Com, Prov, C\#, Tit, Doc, DataE, Voto)}
$$
![[Screenshot 2024-10-10 alle 14.28.28.png|center|550]]
### Problemi
I dati anagrafici di uno studente sono memorizzati per ogni esame sostenuto dallo studente e i dati di un corso sono memorizzati per ogni esame sostenuto per quel corso

La ridondanza dunque dà luogo a:
- spreco di spazio di memoria
- **anomalie** di aggiornamento, inserimento e cancellazione

#### Anomalia di aggiornamento
Se cambia il docente del corso il dato deve essere modificato per ogni esame sostenuto per quel corso
#### Anomalia di inserimento
Non posso inserire i dati anagrafici di uno finché non ha sostenuto almeno un esame a meno che di non usare valori nulli; idem per i corsi
#### Anomalia di cancellazione
Eliminando i dati anagrafici di uno studente potrebbero essere eliminati i dati di un corso (se lo studente è l’unico ad aver sostenuto l’esame di quel corso); idem quando elimino un corso
