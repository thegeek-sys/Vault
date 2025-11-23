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

### Programming language history
At the machine level, data is manipulated by machine instructions executed by the computer processor and is stored either in the processor’s registers or in memory. In assembly language, the programmer is responsible for the correct interpretation of any saved data value, making programming at this level highly precise but also prone to errors.

Languages like C and its derivatives provide high-level control structures, which make programming more convenient, but they still allow direct access to memory. This capability, while powerful, also makes programs written in these languages vulnerable to issues such as buffer overflows. As a result, a large legacy of widely used C code is potentially unsafe and prone to security vulnerabilities.

Modern high-level programming languages, on the other hand, enforce a strong notion of data types and valid operations. This design helps prevent common memory-related vulnerabilities, such as buffer overflows, providing greater safety and reliability. However, these languages often incur additional overhead and impose some limitations on low-level operations compared to languages like C.

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
>
>>[!info] Stack values
>>![[Pasted image 20251122183410.png|500]]

### Common unsafe C standard library routines

| Command                                      | Description                                             |
| -------------------------------------------- | ------------------------------------------------------- |
| `gets(char *str)`                            | read line from standard input into `str`                |
| `sprintf(char⠀*str,⠀char⠀*format,⠀...)`      | create `str` according to supplied format and variables |
| `strcat(char *dest, char *src)`              | append contents of string `src` to string `dest`        |
| `strcpy(char *dest, char *src)`              | copy contents of string `stc` to string `dest`          |
| `vsprintf(char⠀*str,⠀char⠀*fmt,⠀va_list⠀ap)` | create `str` according to supplied format and variables |

---
## Buffer overflow attacks
To exploit a buffer overflow an attacker needs:
- to identify a buffer overflow vulnerability in some program that can be triggered using externally sourced data under the attacker’s control
- to understand how that buffer is stored in memory and determine potential for corruption


Identifying vulnerable programs can be done by:
- inspecting the program source
- tracing the execution of programs as they process oversized input
- using tools such as fuzzing to automatically identify potentially vulnerable programs

### Stack buffer overflows
This kind of overflows occur when buffer is located on stack (also known as stack smashing). It was firstly used by Morris Worm and exploits included an unchecked buffer overflow (but are still being widely exploited).

On the stack frame when one function calls another it needs somewhere to save the return address and needs locations to save the parameters to be passed in to the called function and to possibly save register values

![[Pasted image 20251122223824.png|250]]

### Shellcode
The shellcode is some code supplied by the attacker that is often saved in buffer being overflowed. Usually it consist of transfering control to a command-line interpreter (shell)

While machine code is specific to processor and operating system