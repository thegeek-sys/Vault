---
Created: 2024-11-18
Class: "[[Sistemi Operativi]]"
Related: 
Completed:
---
---
## Cosa contengono?
Le directory contengono le informazioni sui file che sono:
- attributi
- posizione (dove sono i dati)
- proprietario
Una directory è essa stessa un file (speciale) che fornisce un mapping tra nomi dei file e file stessi

---
## Operazioni su una directory
Su una directory si può effettuare operazioni di:
- ricerca
- creazione file
- cancellazione file
- lista del contenuto della directory
- modifica della directory

---
## Elementi delle directory
### Informazioni di base
Le informazioni di base che devono essere da un file system sono:
- Nome del file → nome scelto dal creatore (utente o processo) e unico in una directory data
- Tipo del file → eseguibile, testo, binario, etc.
- Organizzazione del file → per sistemi che supportano diverse possibili organizzazioni (outdated)
### Informazioni sull’indirizzo
- Volume → indica il dispositivo su cui il file è memorizzato
- Indirizzo di partenza → dove si trovano le informazioni (dipende fortemente dal file system, ad es. da quale settore o traccia del disco)
- Dimensione attuale → in byte, word o blocchi
- Dimensione allocata → dimensione massima del file
### Controllo di accesso
- Proprietario → può concedere/negare i permessi ad altri utenti (leggere/scrivere) e può anche cambiare tali impostazioni
- Informazioni sull’accesso → potrebbe contenere username e password per ogni utente autorizzato (chi può leggere scrivere)
- Azioni permesse → per controllare lettura, scrittura, esecuzione, spedizione tramite rete
### Informazioni sull’uso
- Data di creazione
- Identità del creatore
- Data dell’ultimo accesso in lettura
- Data dell’ultimo accesso in scrittura
- Identità dell’ultimo lettore
- Identità dell’ultimo scrittore
- Data dell’ultimo backup
- Uso attuale → lock, azione corrente, etc.

---
## Una semplice struttura per le directory