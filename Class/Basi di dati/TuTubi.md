---
Created: 2025-04-30
Class: "[[Basi di dati]]"
Related:
---
---
## Raffinamento dei requisiti
1. utenti
	1. nome
	2. data di iscrizione
	3. gli utenti si devono poter registrare
	4. gli utenti registrati possono pubblicare video, visualizzare quelli disponibili ed esprimere valutazioni e commenti testuali
2. video
	1. titolo
	2. durata
	3. descrizione
	4. nome del file di memorizzazione
	5. categoria (unica, stringa)
	6. tag (almeno una, stringa)
	7. un video può essere in risposta ad un video esistente (vd. 2)
		1. nessun utente può pubblicare un video in risposta ad un video pubblicato da sé stesso
	8. numero di visualizzazioni (vd. 3)
3. servizio di cronologia
	1. data di visualizzazione