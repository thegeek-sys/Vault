![[52F9FB6A-CCDC-4002-8221-A8EA1ABC1FEF.jpeg]]

Per cui:
- occorrono **2 stalli** tra `li` (i3) e `beq` (i4)
- il ciclo impiega 5 colpi di clock più **2 stalli** tra `lw` (i5) e `add` (i6) e **1 stallo** tra `addi` (i7) e `beq` (i4)
- il control hazard su `beq` (i4) alla fine del ciclo inserisce **2 stalli** (a causa della politica BranchNotTaken)
Totale: $4 \text{[riempimento pipeline]}+3+{\color{\red}{2}}+10\cdot(5+{\color{\red}{3}})+1\text{[uscita ciclo]}+2+{\color{\red}{2}}=94\text{ colpi di clock}$
