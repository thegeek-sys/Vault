---
Class: "[[Cybersecurity]]"
Related:
---
---
## A brief history of some buffer overflow attacks
| Year | Description                                                                                                                                                                   |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1988 | the Morris Internet Worm uses a buffer overflow exploit in "fingerd" as one of its attack mechanisms.                                                                         |
| 1995 | a buffer overflow in NCSA httpd 1.3 was discovered and published on the Bugtraq mailing list by Thomas Lopatic.                                                               |
| 1996 | Aleph One published "Smashing the Stack for Fun and Profit" in Phrack magazine, giving a step by step introduction to exploiting stack-based buffer overflow vulnerabilities. |
| 2001 | the Code Red worm exploits a buffer overflow in Microsoft IIS 5.0.                                                                                                            |
| 2003 | the Slammer worm exploits a buffer overflow in Microsoft SQL Server 2000.                                                                                                     |
| 2004 | the Sasser worm exploits a buffer overflow in Microsoft Windows 2000/XP Local Security Authority Subsystem Service (LSASS).                                                   |

---
## Introduction
Buffer overflow is a very common attack mechanism. Even tho there are many prevention techniques, it is still a major concern in fact there is legacy of buggy code in widely deplyed systems and applications

### Definition
A buffer overflow, also known as buffer overrun and buffer overwrite, is defined in the NIST Glossary of Key Information Security Terms as follows:

>[!quote] Buffer overflow
>”A condition at an interface under which more input can be placed into a
buffer or data holding area than the capacity allocated, overwriting other
information. Attackers exploit such a condition to crash a system or to insert specially
crafted code that allows them to gain control of the system.”

---
## Basics
In practical a buffer overflow is a programming error that happens when a process attempts to store data beyond the limits of a fixed-sized buffer located on the stack, heap, or in the data section of the process (overwrites adjacent memory locations).
These overwritten locations could hold other program variables, parameters, or program control flow data

Consequences:
- corruption of program data
- unexpected transfer of control
- memory access violations
- execution of code chosen by the attacker

>[!example]
>```c
>int main(int argc, char *argv[]) {
>	int valid = false;
>	char str1[8];
>	char str2[8];
>	
>	gets(str2); // reads from stdin
>	next_tag(str1); // overflow of str2
>	if (strncmp(str1, str2, 8) == 0) // compares just the first 8 chars
>		valid = true;
>	printf("buffer1: str1(%s), str2(%s), valid(%d)\n", str1, str2, valid);
>}
>```
>
>```bash
>$ cc -g -o buffer1 buffer1.c
>$ ./buffer1
>START
>buffer1: str1(START), str2(START), valid(1)
>$ ./buffer1
>EVILINPUTVALUE
>buffer1: str1(TVALUE), str2(EVILINPUTVALUE), valid(0)
>$ ./buffer1
>BADINPUTBADINPUT
>buffer1: str1(BADINPUT), str2(BADINPUTBADINPUT), valid(1)
>```