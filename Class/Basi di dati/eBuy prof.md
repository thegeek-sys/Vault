---
Created: 2025-04-29
Class: "[[Basi di dati]]"
Related:
---
---
## Raffinamento dei requisiti
1. Utenti
	1. Nome
	2. Data Registrazione
2. Post/Oggetto
	1. utente venditore
	2. Descrizione
	3. Categoria
	4. prezzo di vendita
		1. per i post asta (vd. 5.4)
	5. metodo di pagamento (bonifico, carta di credito ecc...)
	6. Indicare se oggetto nuovo (vd. 6) o usato (vd. 7)
	7. asta o "compralo subito"
	8. istante di pubblicazione
3. Post Asta  
	1. Prezzo Iniziale (reale ≥ 0)
	2. Ammontare dei rialzi (reale > 0)
	3. istante di scadenza
		1. successivo all'istante di creazione del post (vd. 2.8)
	4. bid piazzati (vd. 5)
	5. il prezzo finale di vendita è calcolabile ed è pari a al prezzo iniziale (vd. 3.1) sommato al numero di rialzi moltiplicato per l'ammontare del singolo rialzo (req 3.2)
	6. l'utente che si aggiudica l'asta è quello che ha offerto l'ultimo bid in ordine cronologico
4. Post "Compralo Subito"
	1. prezzo di vendita (vd. 2.4)
	2. utente acquirente (opzionale vd. 1)
5. Bid (offerta)
	1. il post al quale il bid si riferisce (vd. 3)
	2. istante in cui viene piazzato
		1. deve essere successivo all'istante di pubblicazione del post (vd. 2.8)
		2. deve essere precedente o uguale all'istante di scadenza dell'asta (vd. 3.3)
		3. Non possono essere piazzati due bid distinti da due utenti distinti nello stesso istante
		4. utente offerente (vd. 1)
			1. deve essere diverso da quello che ha creato l'asta (vd. 2.1)
6. Oggetti Nuovi
	1. Garanzia (intero ≥ 2)
7. Oggetti Usati
	1. Garanzia Facoltativa (intero > 0, opzionale) non mettiamo 0 perché appunto se è 0 allora non c'è garanzia
	2. condizioni (un valore tra "ottimo", "buono", "discreto", "da sistemare")

---
## Diagramma UML delle classi