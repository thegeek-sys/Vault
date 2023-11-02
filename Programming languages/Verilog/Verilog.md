## Introduction
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

---
## Moduli
![[Screenshot 2023-11-02 at 23-27-28 PowerPoint Presentation - Verilog-1-Combinational-AM-23-Bellani.pdf.png]]
Un blocco con input e output è chiamato un modulo (porte, multiplexer etc. sono esempi di moduli hardware)
I moduli possono essere di due tipi:
- Comportamentale: descrive cosa fa un modulo
- Strutturale: descrive come è fatto a partire da moduli più semplici

```verilog
module example(input logic a, b, c,
               output logic y);
	assign y = ~a & ~b & ~c | a & ~b & ~c | a & ~b & c;
endmodule
```
- `module/endmodule`: richiesto per iniziare/terminare un modulo
- `example`: nome del modulo
- Operatori:
	- ~ → NOT
	- & → AND
	- | → OR

