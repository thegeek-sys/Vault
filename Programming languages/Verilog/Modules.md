---
Created: 2023-11-02
Programming language: "[[Verilog]]"
Related: 
Completed:
---
---
## Introduction
![[verilog module.png]]
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
Gli operatori Bitwise agiscono sui su segnali single-bit oppure su mutli-bit busses

![[bitwise operator.png]]
```verilog
module gates(input logic [3:0] a, b,
			 output logic [3:0] y1, y2, y3, y4, y5);
/* Five different two-input logic
gates acting on 4 bit busses */
	assign y1 = a & b; // AND
	assign y2 = a | b; // OR
	assign y3 = a ^ b; // XOR
	assign y4 = ~(a & b); // NAND
	assign y5 = ~(a | b); // NOR
endmodule
```

> [!NOTE]
> `a[3:0]` rappresenta un bus a 4 bit denominati dal più significativo al meno significativo `a[3] a[2] a[1] a[0]`. Si può denominare il bus `a[4:1]` oppure `a[0:3]` e usare gli indici di conseguenza

---
## Reduction Operators
Gli operatori di riduzione permettono a input multipli di agire su una singola porta. L’operatore di riduzione esiste per le porte: OR, XOR, NAND, NOR,
e XNOR.
Nota: un input multiplo di XOR esegue la parità: TRUE se un numero dispari di input è TRUE

![[reduction operator.png]]
```verilog
module and8(input logic [7:0] a,
			output logic y);
	assign y = &a;
	// &a is much easier to write than
	// assign y = a[7] & a[6] & a[5] & a[4] &
	//            a[3] & a[2] & a[1] & a[0];
endmodule
```
