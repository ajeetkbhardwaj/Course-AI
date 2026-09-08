
# Big Data Algorithms

What is Online Algorithm ?

Why We must study Online Algorithms ?

In many real world processes requires to make decisions as data arrives, withoud knowing the future like

* Stock trading problem (Stock market decision making)
* Dating / secretary problems (sequential selection)
* Ski rental problem
* Web caching problem
* CPU Cache (memory/paging) Design problem
* Machine learning - Online learning (regret minimization)
* Network routing
* File Systems Problems
* Numerous scheduling and resource allocation problems

Therefore, Online Algorithms are best strategy to make arrival time decisions.

Online vs Offline Algorithms ?

Online Algorithms must commit decisions without access to the future input while Offline algorithms know the entire input sequence.

What is Competative Analysis ?

Consider the input sequence stream be $s = (s_1, s_2, \dots, s_n)$, $C_{\text{opt}}(s)$ be optimal offline cost and $C_A(s)$ be cost of online algorithm (A) Then

1. Deterministic Competitive Ratio(DCR) : An Algorithm (A) is *k-competitive* if for all inputs

$$
C_A(s) \le k\cdot C_{\text{opt}}(s) + O(1)
$$

2. Randomized Competitive Ratio(RCR) : An Algorithm is k-competitive if for all inputs

$$
\mathbb{E}$$C_A(s)$$ \le k\cdot C_{\text{opt}}(s)
$$

Remarks - Goal of competitive analysis was to guarantee performance even in the **worst-case** (adversarial sequences).

1. Bicycle-Rental Problem
   I was in IIT Delhi in 2025 for 6 month, my workplace lab was 2.5 km away from the place of living. So, I have to decide to buy a bicycle or rent because i have to live here for 6 month only. I did some research what could be the cost of retal bicycle per day ? Its around 2.5Rs per minutes by Yulu
2. Secretary/Dating Problem
   Task to choose the best candidate among (n), arriving in random order.
   Constraints

* One-at-a-time arrival
* Must accept/reject immediately
* Rejection is final
* Goal: maximize probability of picking the *best* (rank 1)

Classical Secretary Algorithm

1. Observe first (t = n/e) candidates Then **reject all**
2. From candidate (t+1) onward:
   * Accept the first candidate better than all previous.

What will be the probability of success ?

Let the best candidate appear at position (k). We choose them iff $k > t$ and the best among the first $k-1$ candidates lies within the first $t$ candidates.

So, probability of success is

$$
P_k = \frac1n \cdot \frac{t}{k-1}
$$

and total probability of success will be

$$
P = \sum_{k=t+1}^{n} P_k
\approx \frac{t}{n}(\ln n - \ln t)
$$

Maximizing w.r.t. (t) gives

$$
t = \frac{n}{e}
$$

Hence, the algorithm selects the best candidate with probability ≈ 36.8 %, which is optimal.
