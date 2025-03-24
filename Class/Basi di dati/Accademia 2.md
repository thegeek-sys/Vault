---
Created: 2025-03-24
Class: "[[Basi di dati]]"
Related: 
Completed:
---
---
## Index
- [[#Raffinamento dei requisiti|Raffinamento dei requisiti]]
- [[#Diagramma UML delle classi|Diagramma UML delle classi]]
- [[#Specifica dei tipi di dato|Specifica dei tipi di dato]]
---
## Raffinamento dei requisiti
1. Docenti
	1. Nome (una stringa)
	2. Cognome (una stringa)
	3. Data di nascita (una data)
	4. Luogo di nascita (vd. 2)
	5. Posizione universitaria (tra ricercatore, professore associato, professore ordinario)
	6. Impegni che ha (vd. 5)
2. Luoghi
3. Progetti
	1. Nome (una stringa)
	2. Acronimo (una stringa, univoca)
	3. Data di inizio e di fine
	4. Docenti che vi partecipano (vd. 1)
	5. Ogni progetto è composto da work packages (vd. 4)
4. Work Packages
	1. Progetto cui fa riferimento (vd. 3)
	2. Nome (una stringa)
	3. Data di inizio e data di fine
5. Impegni
	1. Gli impegni sono di diversi tipi:
		1. Assenza
		2. Impegni istituzionali
		3. Impegni progettuali
	2. Giorno in cui avvengono (una data)
	3. Durata (per alcuni impegni va misurata in giorni per altri in ore)
	4. Tipologia di impegno
	5. Motivazione (una stringa)

---
## Diagramma UML delle classi
![[Pasted image 20250324104754.png]]

---
## Specifica dei tipi di dato
- PosizioneUniversitaria → {ricercatore, professore_associato, professore_ordinario}
- InizioFine → (inizio:Data, fine:Data>inizio)
