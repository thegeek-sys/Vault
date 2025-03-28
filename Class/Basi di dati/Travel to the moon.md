## Raffinamento
1. Crociere
	1. Codice (una stringa)
	2. Data di inizio
	3. Data di fine
	4. Nave utilizzata (vd. 2)
	5. Itinerario seguito (vd. 4)
	6. Tipo uno tra
		1. luna di miele, di cui interessa
			1. sottotipo, uno tra:
				1. tradizionali → quelli che prevedono un numero di destinazioni romantiche (vd. 3.4.1) maggiore o uguale al numero di destinazioni divertenti (vd. 3.4.2)
				2. alternative → quelle che non sono tradizionali
		2. per famiglie, di cui interessa
			1. se sono adatte per bambini (un booleano)
2. Requisiti sulle navi
	1. Nome
	2. Gradi di comfort (un intero tra 3 e 5)
	3. Capienza (un intero maggiore di 0)
3. Requisiti sulle destinazioni
	1. Nome
	2. Continente
	3. Posti da vedere (vd. 5)
	4. Tipo, uno tra
		1. Romantico
		2. Divertente
	5. Se è esotica o meno (si trova in un continente diverso dall’Europa)
4. Itinerari
	1. Nome
	2. Sequenza ordinata di elementi di cui interessa:
		1. Destinazione (vd. 3)
		2. Arrivo
			1. Ora
			2. Numero d’ordine del giorno (rispetto alla data di inizio della crociera)
		3. Ripartenza
			1. Ora
			2. Numero d’ordine del giorno (rispetto alla data di inizio della crociera)
5. Posti da vedere
	1. Nome
	2. Descrizione
	3. Orari di apertura (una mappa che associa ad ogni giorno della settimana un insieme di fascia oraria, definita in termini di una coppia di orari)
6. Clienti
	1. Nome
	2. Cognome
	3. Età
	4. Indirizzo
7. Prenotazioni
	1. Istante
	2. Crociera (vd. 1)
	3. Numero di posti prenotati (un intero maggiore di zero)
	4. Cliente  che effettua la prenotazione (vd. 6)

---
## Diagramma delle classi UML

---
## Specifica dei tipi di dato
- DeltaDataOra → {giorno:Intero > 0, ora:Ora}
- CodiceCrociera → secondo standard
- Orari → {giorno_settimana:\[0..7\], }

---
## Specifica di classe
### Itinerario
Un’istanza di questa classe rappresenta un itinerario, cioè una sequenza di destinazione, che può essere seguita dalle crociere
#### Specifica delle operazioni di classe
`durata_g(): Intero >= 0`
- precondizioni → nessuna
- postcondizioni:
	- non modifica il livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `(this, d):arrivo` l’unico link `arrivo` che coinvolge `this`
		- `result` è `(this, d).ora.giorno`

### Destinazione
Un’istanza di questa classe rappresenta una destinazione, ossia un luogo toccato da itinerari di crociere
#### Specifica delle operazioni di classe
`esotica(): Booleano`
- precondizioni → nessuna
- postcondizioni →
	- nessuna modifica al livello estensionale
	- il valore di ritorno `result` è così definito
		- sia `c:Continente` tale che `(this,c):destinazione_continente`
		- `result = true` se e solo se `c.esotico = true`

### Crociera
Un’istanza di questa classe rappresenta una corciera
#### Specifica delle operazioni di classe
`fine(): Data`
- precondizioni → nessuna
- postcondizioni →
	- non modifica il livello estensionale
	- sia `i` il valore dell’attributo `inizio` di `this`
	- sia `it:Itinerario` tale che esiste il link `(this, it):crociera_itinerario`
	- sia `d` il risultato di `it.durata_g()`
	- `result` è uguale a `i+d`

`posti_disponibili(t:DataOra): Intero >= 0`
