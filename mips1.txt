.globl main

.data

main:	
	# il problema di questo codice è che, non avendo inizializzato i valori nei registri, non viene eseguito nulla
	# faccio quindi un programma di testing
	addi $s1,$zero,4 # faccio una somma tra il registro con tutti zero e 4 e l'assegno nel registro s1
	addi $s2,$zero,3
	addi $s3,$zero,9
	addi $s4,$zero,4

	sub $t0,$s1,$s2 # scrivo s1-s2 nella registro temporanea t0
	sub $t1,$s3,$s4 # scrivo s3-s4 nella rewgistro temporanea t1
	add $s0,$t0,$t1 # scrivo t0+t1 nel registro s0

