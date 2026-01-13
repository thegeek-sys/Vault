---
Class: "[[Cybersecurity]]"
Related:
---
---
## Index
- [[#Classic DoS attacks|Classic DoS attacks]]
	- [[#Classic DoS attacks#Flooding ping|Flooding ping]]
	- [[#Classic DoS attacks#Source address spoofing|Source address spoofing]]
	- [[#Classic DoS attacks#SYN spoofing|SYN spoofing]]
- [[#Flooding attacks|Flooding attacks]]
- [[#Distributed Denial of Service (DDoS) attacks|Distributed Denial of Service (DDoS) attacks]]
	- [[#Distributed Denial of Service (DDoS) attacks#Mirai short story|Mirai short story]]
	- [[#Distributed Denial of Service (DDoS) attacks#Examples of DDoS attacks|Examples of DDoS attacks]]
- [[#SIP invite scenario|SIP invite scenario]]
- [[#Hypertext Transfer Protocol (HTTP) based attacks|Hypertext Transfer Protocol (HTTP) based attacks]]
	- [[#Hypertext Transfer Protocol (HTTP) based attacks#HTTP flood|HTTP flood]]
	- [[#Hypertext Transfer Protocol (HTTP) based attacks#Slowloris - R.U.D.Y.|Slowloris - R.U.D.Y.]]
- [[#Reflection attacks|Reflection attacks]]
	- [[#Reflection attacks#DNS amplification attacks|DNS amplification attacks]]
- [[#Memcached DDoS attack|Memcached DDoS attack]]
- [[#DoS attack defenses|DoS attack defenses]]
	- [[#DoS attack defenses#Prevention|Prevention]]
	- [[#DoS attack defenses#Responding to DoS attack|Responding to DoS attack]]
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

![[Pasted image 20251125215750.png|420]]

---
## Flooding attacks
Flooding attacks are classified based on the network protocol used. The intent is to overload the network capacity on some link to a server. Virtually any type of network packet can be used.

There are three kinds of this attack:
- ICMP flood → ping flood using ICMP echo request packets as traditionally network administrators allow such packets into their networks because ping is a useful network diagnostic tool
- UDP flood → uses UDP packets directed to some port number on the target system
- TCP SYN flood → sends TCP packets to the target system; total volume of packets is the aim of the attacker rather than the system code

---
## Distributed Denial of Service (DDoS) attacks
DDoS attacks consist in the use of multiple systems to generate attacks. The attacker uses a flaw in operating system or in common application to gain access and install their program on it (zombie)

Large collection of such systems under the control of one attacker can be created, forming a *botnet*

![[Pasted image 20251125221608.png|470]]

### Mirai short story
In September 2016, the authors of the Mirai malware launched a DDoS attack on the website of a well-known security expert ([krebsonsecurity.com](https://krebsonsecurity.com)). A week later they released the source code into the world, possibly in an attempt to hide the origins of the attack

The code was quickly replicated by other cybercriminals, and is believed to be behind the massive attack that brought down the domain registration services provider, Dyn, in October 2016

### Examples of DDoS attacks
The biggest DDoS attack to date happened in September 2017, when the Google services received an attack with size of 2.54 Tbps. It was a reflection attack: spoofed packets sent to 180.000 web servers, which in turn sent responses to Google

In February 2020, AWS saw incoming traffic at a rate of 2.3 Tbps. The attacker responsible used hijacked Connection-less Lightweight Directory Access Protocol (CLDAP) web servers

In February 2018, an attack against Github reached 1.3 Tbps, sending packets at a rate of 126.9 million per second

In May 2025, Cloudflare mitigated the largest distributed denial-of-service (DDoS) attack ever reported, at 7.3 Tbps. The attack lasted around 45 seconds, and in that time delivered about 37.4 Tb of attack traffic
The 7.3 Tbps attack used multiple attack vectors against its target, including UDP floods, NTP reflection attacks, and traffic from the Mirai botnet. Fortunately, the Cloudflare network was able to automatically mitigate the attack and keep the target, a hosting provider, from falling victim.

---
## SIP invite scenario
![[Pasted image 20251125222542.png|400]]

---
## Hypertext Transfer Protocol (HTTP) based attacks
### HTTP flood
HTTP flood is an attack that bombards Web servers with HTTP requests (LOIC, HOIC) and consumes considerable resources

We talk about *spidering* when bots starts from a given HTTP link and by follows all links on the provided website in a recursive way

### Slowloris - R.U.D.Y.
Slowloris attack attempts to monopolize a website by sending HTTP requests that never complete and eventually consumes web server’s connection capacity.

It uses legitimate HTTP traffic as existing intrusion detection and prevention solution rely on signatures to detect attacks, so they generally won’t recognize Slowloris

---
## Reflection attacks
In this case the attacker sends packets to a known service on the intermediary with a spoofed source address of the actual target system and when the intermediary responds, the response is sent to the target, “reflecting” the attack off the intermediary (reflector)

The goal is to generate enough volumes of packets to flood the link to the target system without alerting the intermediary. The basic defense against these attacks is blocking spoofed-source packets

>[!example] DNS reflection attack
>![[Pasted image 20251125223515.png|500]]

### DNS amplification attacks
Amplification attacks use packets directed at a legitimate DNS server as the intermediary system. The attacker creates a series of DNS requests containing the spoofed source address of the target system and exploits the DNS behavior to convert a small request to a much larger response (amplification), so the target is flooded with responses.

![[Pasted image 20251125223822.png|500]]

Basic defense against this attack is to prevent the use of spoofed source address

>[!example] Amplification example
>Request: 64 bytes
>```
>dig ANY isc.org @x.x.x.x
>```
>
>Which results in a 50x amplification

---
## Memcached DDoS attack
![[Pasted image 20251125230530.png|400]]

Memcached is a high-performance caching mechanism for dynamic websites, that allows to speed up the delivery of web contents. The idea is to make a request that stores a large amount of data and than send a spoofed request to make such data to be delivered to the victim via UDP (connection-less)

>[!tip]
>Memcached DDoS attack can bring an amplification factor of 50000!

---
## DoS attack defenses
These attacks cannot be prevented entirely as high traffic volumes may be legitimate. There are four lines of defense against DDoS attacks:
- attack prevention and preemption → before the attack
- attack detection and filtering → during the attack
- attack source traceback and identification → during and after the attack
- attack reaction → after the attack

### Prevention
- block spoofed source address (on routers as close to source as possible)
- filters may be used to ensure path back to the claimed source address is the one being used by the current packet; filters must be applied to traffic before it leaves the ISP’s network or at the point of entry to their network
- use modified TCP connection handling code; cryptographically encode critical information in a cookie that is send as the server’s initial sequence number (legitimate client responds with an ACK packet containing the incremented sequence number cookie) and drop an entry for an incomplete connection from the TCP connections table when it overflows
- block IP directed broadcast
- block suspicious services and combinations
- manage application attacks with a form of graphical puzzle (captcha) to distinguish legitimate human requests
- good general system security practices
- use mirrored and replicated servers when high-performance and reliability is required

### Responding to DoS attack

>[!info] Good incident response plan
>- details on how to contact technical personal for ISP
>- needed to impose traffic filtering upstream
>- details of how to respond to the attack

- antispoofing, directed broadcast, and rate limiting filters should have been implemented
- ideally have network monitors and IDS to detect and notify abnormal traffic patterns
- **identify type of attack** → capture and analyze packets, design filters to block attack traffic upstream, or identify and correct system/application bug
- **have ISP trace packet flow back to source** → may be difficult and time consuming but necessary if planning legal action
- **implement contingency plan** → witch to alternate backup servers and commission new servers at a new site with new addresses
- **update incident response plan** → analyze the attack and the response for future handling