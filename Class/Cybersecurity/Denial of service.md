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
The aim of this attack is to overwhelm the capacity of the network connection to the target organization, in particular, even tho