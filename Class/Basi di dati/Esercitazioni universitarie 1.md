---
Created: 2025-03-29
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Raffinamento dei requisiti
1. Insegnamenti
	1. nome (una stringa)
	2. crediti formativi (un intero > 0)
	3. tenuto da uno o più docenti (vd. 2)
	4. può prevedere diverse esercitazioni (vd. 3)
2. Docenti
	1. Matricola (un intero)
	2. Nome (una stringa)
	3. Cognome (una stringa)
3. Esercitazioni
	1. Data (una data)
	2. Esercizi (vd. 4)
4. Esercizi
	1. Testo (una stringa)
	2. soluzioni disponibili 
	3. ogni esercizio puo essere:
		1. solo presentato (svolto in modo autonomo)
		2. risolto in aula (quale delle soluzioni disponibili è stata mostrata)

---
## Diagramma UML delle classi
![[Pasted image 20250401092945.png]]

---
## Specifica sui tipi di dato
- Matricola → secondo standard