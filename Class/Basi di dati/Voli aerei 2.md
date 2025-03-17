---
Created: 2025-03-17
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Obiettivi
Si vuole sviluppare un sistema informativo per la gestione di dati relativi a voli aerei.
Durante la fase di raccolta dei requisiti è stata prodotta la seguente specifica dei requisiti.
Si chiede di iniziare la fase di Analisi Concettuale ed in particolare di:
1. raffinare la specifica dei requisiti eliminando inconsistenze, omissioni o ridondanze e produrre un elenco numerato di requisiti il meno ambiguo possibile
2. produrre un diagramma UML delle classi concettuale che modelli i dati di interesse, utilizzando solo i costrutti di classe, associazione, attributo, generalizzazione tra classi
3. produrre la relativa specifica dei tipi di dato in caso si siano definiti nuovi tipi di dato concettuali.

---
## Specifica dei requisiti
I dati di interesse per il sistema sono voli, compagnie aeree ed aeroporti.
Dei voli interessa rappresentare codice, durata, compagnia aerea ed aeroporti di partenza e arrivo.
Degli aeroporti interessa rappresentare codice, nome, città (con nome e numero di abitanti) e nazione.
Delle compagnie aeree interessa rappresentare nome, anno di fondazione, e la città in cui ha sede la direzione.
Un tipo particolare di voli sono voli charter. Questi possono prevedere tappe intermedie in aeroporti. Delle tappe intermedie di un volo charter interessa mantenere l’ordine con cui esse si susseguono (ad esempio, un certo volo che parte da “Milano Linate” e arriva a “Palermo Punta Raisi”, prevede tappe intermedie prima nell’aeroporto di Bologna e poi in quello di Napoli). Dei voli charter interessa rappresentare anche il modello di velivolo usato

---
## Raffinamento dei requisiti
1. Voli
    1. Codice (una stringa)
    2. Durata (un intero)
    3. Compagnia aerea (vd. 2)
    4. Aeroporto partenza (vd. 4)
    5. Aeroporto arrivo (vd. 4)
    6. Voli charter (tipo particolare di volo)
        1. Possono avere tappe intermedie (vd. 2)
        2. Modello velivolo (una stringa)
2. Aeroporti
    1. Codice (una stringa)
    2. Nome (una stringa)
    3. Città (vd. 3)
3. Città
    1. Nome (una stringa)
    2. Abitanti (un intero)
    3. Nazione (vd. 5)
4. Compagnie aeree
    4. Nome (una stringa)
    5. Anno di findazione (una stringa)
    6. Città della sede (vd. 3)
5. Nazione
    1. Nome (una stringa)

---
## Diagramma UML delle classi
![[Pasted image 20250317094328.png]]

---
## Specifica dei tipi di dato
- CodiceVolo → come da standard
- CodiceAeroporto → come da standard
