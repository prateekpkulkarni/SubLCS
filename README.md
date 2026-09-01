# Subsystem Logical Clifford Synthesis (SubLCS)

Reference implementation of Subsystem Logical Clifford Synthesis (SubLCS), providing complete enumeration of all physical Clifford realizations of a logical Clifford operator on subsystem stabilizer codes.

📄 Paper: [Logical Clifford Synthesis for Subsystem Codes](https://github.com/prateekpkulkarni/SubLCS/blob/main/SubsystemLCS_Gauge.pdf) 
💻 Code: [sublcs.py](https://github.com/prateekpkulkarni/SubLCS/blob/main/sublcs.py)

---

## Overview

This repository implements SubLCS, a generalization of the Logical Clifford Synthesis (LCS) framework of Rengaswamy et al. (IEEE TQE, 2020) from stabilizer codes to subsystem stabilizer codes.

In the original LCS framework, all valid physical implementations of a logical Clifford operator are derived using stabilizer structure. However, subsystem codes introduce additional degrees of freedom through the gauge group. These degrees of freedom are not captured by stabilizer-only theory and significantly expand the space of valid implementations.

This work shows that the full solution space naturally separates into:
- a stabilizer-driven component (handled by classical LCS), and  
- a gauge-driven component arising from additional symmetries in subsystem codes.

This leads to a complete and exact enumeration of all valid physical realizations, while also enabling independent optimization over stabilizer and gauge choices.

For full details, proofs, and formal development, see the paper above.

---

## Algorithm

The implementation follows **Algorithm 1 (SubLCS)** from the paper.

At a high level:
- Phase 1 enumerates all stabilizer-consistent implementations using the original LCS method.
- Phase 2 explores additional implementations enabled by gauge freedom.
- The final solution space is obtained by combining both components.

👉 **Code reference:**  
The implementation of the **Phase 2 (gauge orbit construction and lifting)** — the key extension beyond classical LCS — can be found here:

➡️ [`sublcs.py`](https://github.com/prateekpkulkarni/SubLCS/blob/main/sublcs.py)

---

## Usage

```bash
python sublcs.py
````

---

## Example

The implementation reproduces the [[4,1,1,2]] Bacon–Shor subsystem code:

* Stabilizer-only synthesis produces a limited set of implementations
* Incorporating gauge transformations expands this to the full solution space

This demonstrates the additional flexibility provided by subsystem codes.

---

## Output

Each solution is represented as a symplectic matrix corresponding to a valid physical Clifford circuit.

These can be converted into explicit quantum circuits using standard decomposition techniques.

---

## References

* N. Rengaswamy, R. Calderbank, S. Kadhe, H. D. Pfister.
  *Logical Clifford Synthesis for Stabilizer Codes*, IEEE TQE, 2020

* P. P. Kulkarni.
  *Logical Clifford Synthesis for Subsystem Codes* (Under Submission, 2026)

---

## Remarks

* Works for arbitrary subsystem codes given stabilizer, gauge, and logical descriptions
* Full enumeration becomes expensive for large gauge dimensions
* Intended as a reference implementation of the SubLCS framework

---

