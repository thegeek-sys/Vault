---
Created: 2024-03-05
Programming language: "[[Java]]"
Related: 
Completed:
---
---
## Index

- [[#Introduzione|Introduzione]]
- [[#`private, public`|private, public]]
- [[#Accesso a campi e metodi|Accesso a campi e metodi]]
---
## Introduzione
*Perché utilizzare le keyword public e private?*
Perché ci sono delle informazioni che devono essere nascoste all’utente (“information hiding“). Il processo che nasconde i dettagli realizzativi (campi e implementazione), rendendo privata un’interfaccia (metodi pubblici), prende il nome di incapsulamento.

Questo processo ci può essere utile per semplificare e modularizzare il lavoro di sviluppo assumendo un certo funzionamento a “scatola nera”. Non è necessario sapere tutto. L’incapsulamento **facilita il lavoro di gruppo e l’aggiornamento del codice**.
Una classe interagisce con e altre principalmente attraverso i costruttori e metodi pubblici. Le altre classi non devono conoscere i dettagli implementativi di una classe per utilizzarla in modo efficace

---
## `private, public`
Questi due sono modificatori della visibilità. Questi permettono ad una classe o metodo di essere accessibile da altre classi.
Una classe **pubblica** può essere visibile da ogni altra classe (indipendentemente da il package in cui si trova). Un campo **privato** invece non può essere accessibile da altre classi (neanche nello stesso package)

---
## Accesso a campi e metodi
I campi e i metodi possono essere pubblici o privati. I metodi di una classe possono chiamare i metodi pubblici e privati della stessa classe. I metodi di una classe possono chiamare i metodi pubblici (ma non quelli privati) di altre classi