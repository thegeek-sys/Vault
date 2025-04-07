---
Created: 2025-04-07
Class: "[[Basi di dati]]"
Related:
---
---
## Raffinamento dei requisiti
1. Officine della catena
	1. nome (una stringa)
	2. indirizzo (una stringa)
	3. numero di dipendenti (un intero)
	4. dipendenti (vd. 2)
		1. numero di anni di servizio (un intero)
2. Dipendenti
	1. nome (una stringa)
	2. codice fiscale (una stringa)
	3. indirizzo (una stringa)
	4. numero di telefono (una stringa)
	5. Direttori
		1. dei direttori interessa anche la data di nascita
3. Riparazioni dei veicoli
	1. codice (un intero)
	2. veicolo (modello, tipo, targa, anno di immatricolazione e proprietario)
	3. data e ora di accettazione
	4. data e ora di riconsegna (per le riparazione terminate)
4. Proprietario
	1. nome (una stringa)
	2. codice fiscale (una stringa)
	3. indirizzo
	4. telefono

---
## Diagramma UML della classi
![[Pasted image 20250407134910.png]]

---
## Specifica dei tipi di dato
- CodiceFiscale → secondo standard
- Telefono → secondo standard

---
## Specifica di classe
### Officina
Un’istanza di questa classe rappresenta un’officina della catena
#### Specifica delle operazioni di classe
`numero_dipendenti(): Intero >= 0`
- precondizioni → nessuna
- post condizioni
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `d:Dipendente` tale che esista il link `(this,d):off_dip`
		- sia `D` l’insieme dei link `off_dip`
		- `result` è la cardinalità di `D`

### Dipendente
Un’istanza di questa classe rappresenta un dipendente dell’officina
#### Specifica delle operazioni di classe
`anni_servizio(): Intero >= 0`
- precondizioni → sia `o:Officina` tale che esista l’unico link `(this, o):off_dip`
- postcondizioni
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `a=off_dip.anno_assunzione`
		- `result=adesso.anno-a`

### Riparazione
Un’istanza di questa classe rappresenta una riparazione dell’officina
#### Specifica delle operazioni di classe
`fine_riparazione(): Data`
- precondizioni → sia `vei:Veicolo` tale che esista l’unico link `(this, vei):fine_rip`
- postcondizioni → 
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `v:velicolo` tale che esista il link `(this, v):rip_vei`
		- sia `f=fine_rip.durata_giorni` e sia `i=rip_vei.data`
		- `result=i+f`
