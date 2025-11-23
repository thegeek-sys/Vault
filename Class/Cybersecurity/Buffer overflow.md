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

### Common x86 Assembly language instructions

| Instruction            | Description                                                            |
| ---------------------- | ---------------------------------------------------------------------- |
| `MOV src, dest`        | copy (move) value from `src` into `dest`                               |
| `LEA src, dest`        | copy the address (load effective address) of `src` into `dest`         |
| `ADD/SUB src, dest`    | add/sub value in `src` from dest leaving result in `dest`              |
| `AND/OR/XOR⠀src,⠀dest` | logical and/or/xor value in `src` with `dest` leaving result in `dest` |
| `CMP val1, val2`       | compare `val1` and `val2`, setting CPU flags as a result               |
| `JMP/JZ/JNZ addr`      | jump/if zero/if not zero to addr                                       |
| `PUSH src`             | push the value in `src` onto the stack                                 |
| `POP dest`             | pop the value on the top of the stack into `dest`                      |
| `CALL addr`            | call function at `addr`                                                |
| `LEAVE`                | clean up stack frame before leaving function                           |
| `RET`                  | return from function                                                   |
| `INT num`              | software interrupt to access operating system function                 |
| `NOP`                  | no operation or do nothing instruction                                 |

### x86 registers
| 32 bit | 16 bit | 8 bit (high) | 8 bit (low) | Use                                                                                                   |
| ------ | ------ | ------------ | ----------- | ----------------------------------------------------------------------------------------------------- |
| `%eax` | `%ax`  | `%ah`        | `%al`       | Accumulators used for arithmetical and I/O operations and execute interrupt calls                     |
| `%ebx` | `%bx`  | `%bh`        | `%bl`       | Base registers used to access memory, pass system call arguments and return values                    |
| `%ecx` | `%cx`  | `%ch`        | `%cl`       | Counter registers (often used for loop counts)                                                        |
| `%edx` | `%dx`  | `%dh`        | `%dl`       | Data registers used for arithmetic operations, interrupt calls and I/O operations                     |
| `%ebp` |        |              |             | Base Pointer containing the address of the current stack frame                                        |
| `%eip` |        |              |             | Instruction Pointer or Program Counter containing the address of the next instruction to be executed5 |
| `%esi` |        |              |             | Source Index register used as a pointer for string or array operations                                |
| `%esp` |        |              |             | Stack Pointer containing the address of the top of stack                                              |

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

Shellcode is written in machine code and is therefore specific to both the processor architecture and the operating system. Classic examples include code that performs the equivalent of invoking a UNIX shell using a system call like `execve("/bin/sh")`, or launching a command interpreter such as `"command.exe"` on Windows systems. Historically, producing such code required strong assembly-language skills due to the low-level and architecture-dependent nature of the work.

More recently a number of sites and tools have been developed to automate this process. A prime example is the Metasploit Project that provides useful information to people who perform penetration, IDS signature development, and exploit research

>[!example]
>```c
>int main(int argc, char *argv[]) {
>	char *sh;*
>	char *args[2];
>	
>	sh = "/bin/sh";
>	args[0] = sh;
>	args[1] = NULL;
>	execve(sh, args, NULL);
>}
>```
>
>>[!info] Equivalent position-indipendent x86 assembly code
>>```js
>>       nop
>>       nop                 //end of nop sled
>>       jmp find            //jump to end of code
>>cont: pop %esi            //pop address of sh off stack into esi
>>       xor %eax, %eax      //zero contents of EAX
>>       mov %al, 0x7(%esi)  //copy zero byte to end of string sh esi
>>       lea (%esi), %ebx    //load address of sh (%esi) into %ebx
>>       mov %ebx, 0x8(%esi) //save address of sh in args [0] (esi+8)
>>       mov %eax, 0xc(%esi) //copy zero to args[1] (%esi+c)
>>       mov $0xb, %al       //copy execve syscall number (11) to AL
>>       mov %esi, %ebx      //copy address of sh (%esi) into ebx
>>       lea 0x8(%esi), %ecx //copy address of args[0] (esi+8) to ecx
>>       lea 0xc(%esi), %edx //copy address of args[1] (esi+c) to edx
>>       int $0x80           //software interrupt to execute syscall
>>find: call cont           //call cont that saves next addr onstack
>>sh:   .string "/bin/sh "  //string constant
>>args: .long 0             //space used for args array
>>       .long 0             //args[1] and also NULL for env array
>>```
>
>![[Pasted image 20251123154604.png]]

>[!warning] Shellcode caveats
>It has to be position independent, so the shellcode must be able to run no matter where in memory it is located. The attacker in fact generally cannot determine in advance exactly where the targeted buffer will be located in the stack frame of the function in which it is defined, so only relative address references can be used (the attacker is not able to precisely specify the starting address of the instructions in the shellcode).
>
>It cannot contain any `NULL` values in fact it uses unsafe string manipulation routines and strings end with `NULL` values

---
## Stack overflow variants
Target programs can be:
- trusted system utility
- network service daemon
- commonly used library code

An example are shellcode functions. Those can launch a remote shell when an attacker connect to it, creating a reverse shell that connects back to the hacker. It uses local exploits that establish a shell and flushed firewall rules that currently block other attacks

---
## Bufferoverflow defenses
Buffer overflows are widely exploited so there are many ways to protect against them. They divide into two approaches:
- compile-time → aims to harden programs to resist attacks in new programs
- run-time → aim to detect and abort attacks in existing programs

### Compile time defenses
The safest way is to use a modern high-level language not vulnerable to overflow attacks and a compiler that enforces range checks and permissible operations on variables

#### Disadvantages
- additional code must be executed at run time to impose checks
- flexibility and safety comes at a cost in resource use
- distance from the underlying machine language and architecture means that access to some instructions and hardware resources is lost
- limits their usefulness in writing code, such as device drivers, that must interact with such resources

### Compile-Time Defenses
