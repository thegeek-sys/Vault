---
Created: 2025-03-29
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Raffinamento dei requisiti
1. Giocatori
	1. nickname (una stringa, univoca)
	2. nome (una stringa)
	3. cognome (una stringa)
	4. indirizzo (una stringa)
	5. rank (intero positiv > 0)
2. Partite
	1. Data (una data)
	2. Luogo in cui è giocata (una stringa?)
	3. Regole di conteggio (cinese o giapponese)
	4. Quale giocatore gioca con pietre bianche e quale con le nere (vd. 1)
	5. komi (un reale non negativo tra 0 e 10)
	6. esito (vd. 3)
	7. le partite si possono riferire a un torneo (vd. 4)
3. Esito partite può essere:
	1. Rinuncia da parte di uno dei due giocatori
	2. Coppia di punteggi di bianco e di nero (interi non negativi)
4. Tornei
	1. Nome (una stringa)
	2. Desctizione testuale (una stringa)
	3. edizione (un intero, riferisce un anno)