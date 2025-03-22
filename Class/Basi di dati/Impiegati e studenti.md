---
Created: 2025-03-22
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Raffinamento dei requisiti
1. **Persone**
	1. Nome (una stringa)
	2. Cognome (una stringa)
	3. Codice fiscale (secondo standard, univoca)
	4. Data di nascita (una data)
	5. Degli **uomini** interessa anche
		1. Posizione militare (vd. 2)
	6. Delle **donne** interessa anche
		1. Numero di maternità (un intero ≥ 0)
	7. Alcune persone sono **impiegati** (vd. 3)
	8. Alcune persone sono **studenti** (vd. 6)
2. **Posizioni militari**
	1. Nome (una stringa)
3. **Impiegati**
	1. Stipendio (un intero ≥ 0)
	2. Ruolo (può essere **segretario**, **direttore** o **progettista** vd.4 )
4. **Progettisti**
	1. Progetto/i (vd.5) di cui sono responsabili (anche nessuno)
5. **Progetti**
	1. Nome (una stringa)
6. **Studenti**
	1. Numero di matricola (un intero ≥ 0, univoco)

