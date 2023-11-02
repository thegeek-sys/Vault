---
Created: 2023-11-02
Programming language: "[[Verilog]]"
Related: 
Completed:
---
---
## Introduction
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
- Definisco nomi e tipi di variabili di input e output
- Operatori:
	- ~ → NOT
	- & → AND
	- | → OR
	- ^ → XOR

---
## Modellamento strutturale - gerarchia
In questo programmino sottostante ho definito 3 moduli in totale
1. Che esegue un and tra tre variabili
2. Che nega una variabile
3. Che definisce un segnale interno (una sorta di variabile locale) che poi viene utilizzata come output del primo modulo ed input del secondo

```verilog
module and3(input logic a, b ,c
		    output logic y);
	assign y = a & b & c;
endmodule
---------
module inv(input logic a
		    output logic y);
	assign y = ~a;
endmodule
---------
module nand3(input logic a, b ,c
		    output logic y);
	logic n1;                  // segnale interno
							   // sono usati solo all'interno del
							   // modulo stesso (simili alle
							   // variabili locali)

	and3 andgate(a, b, c, n1)  // istanza di and3
	inv inverter(n1, y)        // istanza di inv
endmodule
```

---
## Bitwise Operators
