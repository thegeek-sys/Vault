---
Created: 2023-11-02
Programming language: "[[Verilog]]"
Related: 
Completed:
---
---
Verilog è un **Hardware description language** (HDL) ovvero un linguaggio il cui compito è solo quello di definire delle funzioni logiche. Tra i maggiori HDLs troviamo:
- SystemVerilog
- VHDL 2008

Gli HDL possono avere due possibili applicazioni:
- Implementazione
	- Trasforma il codice HDL in una *netlist* che descrive l’hardware (es. una lista di porte e i cavi che le connettono)
	- Il sintetizzatore logico potrebbe eseguire delle operazioni per ottimizzare e ridurre la quantità di hardware richiesto
	- La netlist potrebbe essere un file di testo, oppure può essere disegnato in modo schematico per aiutare a visualizzare il circuito
- Simulazione
	- Vengono applicati degli inputo ai circuiti e viene verificata la correttezza degli output

> [!WARNING]
> Quando si usa un HDL bisogna pensare all’hardware che deve produrre
