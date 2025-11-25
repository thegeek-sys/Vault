---
Class: "[[Cybersecurity]]"
Related:
---
---
## Introduction

>[!quote] NIST Computer Security Incident Handling Guide definition of a DoS attack
>”An action that prevents or impairs the authorized use of networks, systems, or applications by exhausting resources such as central processing units (CPU), memory, bandwidth, and disk space.”

Denial-of-Service (DoS) is a form of attack on the availability of some service. The categories of resources that could be attacked are:
- *network* (bandwidth) → relates to the capacity of the network links connecting a server to the network; for most organizations this is their connection to their ISP
- *system* (resources) → aims to overload or crash the network handling software
- *application* (resources) → typically involves a number of valid requests, each of which consumes significant resources, thus limiting the ability of the server to respond to requests from other users

>[!example]
>![[Pasted image 20251125213527.png|450]]

---
## Classic DoS attacks
### Flooding ping
The aim of this attack is to overwhelm the capacity of the network connection to the target organization, in particular, even tho this traffic can be handled by higher capacity links on the path, as they reach links with lower capacity the packets are discarded and the network performance is noticeably affected.

Source of the attack is clearly identified unless a spoofed address is used

### Source address spoofing
Attacker use a forged source address, usually via the raw socket interface on operating systems to make the attack harder to identify.
Then the attacker generates large volumes of packets that have the target system as the destination address (but different origin address).

Congestion would result in the router connected to the final, lower capacity link, and this attack will require network engineers to specifically query flow information from their routers

### SYN spoofing
This one of the most common DoS attack. It attacks the ability of a server to respond to future connection requests by overflowing the tables used to manage them (legitimate users are denied access to the server).

Hence an attack on system resources, specifically the network handling code in operating system