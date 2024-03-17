---
Created: 2024-03-17
Class: "[[Architettura degli elaboratori]]"
Related: 
Completed:
---
---
## Introduction
Per allocare una stringa in memoria, quello che facciamo altro non è che creare un vettore di byte, in cui, in ogni byte è memorizzata un carattere codificato secondo lo standard ASCII.
C’è però da notare che alla fine della stringa verrà salvato un carattere di fine stringa `\0` che nonostante rappresenti un carattere vuoto, è diverso dal carattere di spaziatura (`blank`)

Per esempio:
`label: .asciiz "sopra la panca"`
![[Screenshot 2024-03-17 alle 17.19.02.png]]

---
## American Standard Code for Information Interchange (ASCII)
![[Screenshot 2024-03-17 alle 17.22.54.png]]
https://theasciicode.com.ar

---
## Endianess
Il concetto di **endianess** (se un dato sistema è little endian o big endian) ha a che fare con:
- indirizzamento in memoria (che avviene per byte, ogni indirizzo specifica il singolo byte)
- come i byte di una word sono ordinati in memoria

Il processore MIPS permette l’ordinamento dei byte di una word in due modi:
- **Big-endian** (o network-order, usato da Java e dalle CPU SPARC Sun/Oracle)
	i byte della word sono memorizzati **dal most** significant byte **al least** significant byte
	![[Screenshot 2024-03-17 alle 17.30.01.png]]
	
- **Little-endian** (usato dalle CPU Intel, ad es. Windows, e da MARS)
	i byte della word sono memorizzati **dal least** significant byte **al most** significant byte
	![[Screenshot 2024-03-17 alle 17.31.27.png]]

### Esempio
`label: .asciiz "sopra la panca."`
![[Screenshot 2024-03-17 alle 17.33.49.png|500]]

`label: .word 0x6369616F  # "ciao" in hex`
![[Screenshot 2024-03-17 alle 17.52.22.png|150]]
