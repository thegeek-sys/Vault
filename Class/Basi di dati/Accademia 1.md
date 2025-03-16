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

