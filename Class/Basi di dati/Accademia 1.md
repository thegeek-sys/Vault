---
Created: 2025-03-16
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Obiettivi
Si vuole progettare un sistema informativo per la gestione delle tabelle orarie relative al ruolo di docente universitario. Durante la fase di raccolta dei requisiti è stata prodotta la seguente specifica dei requisiti.

Si chiede di iniziare la fase di Analisi Concettuale ed in particolare di:
1. raffinare la specifica dei requisiti eliminando inconsistenze, omissioni o ridondanze e produrre un elenco numerato di requisiti il meno ambiguo possibile
2. produrre un diagramma UML delle classi concettuale che modelli i dati di interesse, utilizzando solo i costrutti di classe, associazione, attributo
3. produrre la relativa specifica dei tipi di dato in caso si siano definiti nuovi tipi di dato concettuali.

---
## Specifica dei requisiti
I dati di interesse per il sistema sono i docenti universitari, i progetti di ricerca e le attività dei docenti.
Di ogni docente interessa conoscere il nome, il cognome, la data di nascita, la matricola, la posizione universitaria (ricercatore, professore associato, professore ordinario) e i progetti ai quali partecipa.
Dei progetti interessa il nome, un acronimo, la data di inizio, la data di fine e i docenti che vi partecipano.
Un progetto è composto da molti Work Package (WP). Oltre al progetto a cui fa riferimento, del WP interessa sapere il nome, la data di inizio e la data di fine.
Il sistema deve permettere ai docenti di registrare impegni di diverso tipo. Degli impegni interessa sapere il giorno in cui avvengono, la durata in ore e la tipologia di impegno con relativa motivazione.

---
## Raffinamento dei requisiti
1. Docenti universitari
    1. Nome (una stringa)
    2. Cognome (una stringa)
    3. Data di nascita (una data)
    4. Matricola (un intero)
    5. Posizione universitaria tra:
        1. Ricercatore
        2. Professore associato
        3. Professore ordinario
    6. Progetti a cui partecipa (vd. 2)
    7. Impegni che ha (vd. 4)
2. Progetti
    8. Nome (una stringa)
    9. Acronimo (una stringa)
    10. Data di inizio (una data)
    11. Data di fine (una data)
    12. Docenti che vi partecipano
    13. molti Work Package (vd. 3)
3. Work Package
    1. Nome (una stringa)
    2. Data di inizio (una data)
    3. Data di fine (una data)
4. Imepgni
    4. Giorno in cui avvengono (una data)
    5. Durata in ore (un intero)
    6. Tipologia di impegno (vd. 5)
5. Tipologia di impegno
    1. Motivazione (una stringa)

---
## Diagramma UML delle classi
![[Pasted image 20250316194223.png]]

---
## Specifica dei tipi di dato
- PosizioneUniversitaria → {ricercatore, professore_associato, professore_ordinario}
- InizioFine → (inizio:Data, fine:Data>inizio)