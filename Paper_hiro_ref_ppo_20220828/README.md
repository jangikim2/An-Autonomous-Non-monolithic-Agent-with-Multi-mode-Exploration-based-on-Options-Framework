
# Implementation: Data-Efficient Hierarchical Reinforcement Learning

Implementation practice of RL project (the second practicing project).

- [x] It's better to browse this markdown in **[Typora](https://typora.io/)** for better math rendering.
- [x] This Repository has been included by [paperswithcode.com](https://paperswithcode.com/paper/data-efficient-hierarchical-reinforcement)!

### Contents:

* [Intro to HIRO](#intro-to-hiro)
* Understand **[value-based RL](valueRL.md)** methods
* [Implementation structure](#implementation-structure)
* [Start to use this code base](#start-to-use-this-code-base)
* [Experiment results](#experiment-results)

## Intro to HIRO

Original Paper: [Data Efficient HRL](https://arxiv.org/abs/1805.08296)

### General Intro

HIRO represents "HIerarchical Reinforcement learning with Off-policy correction". The motivation of this paper is to train both HRL low-level policy and high-level policy with off-policy experience. The authors think this will help improve data efficiency and makes the method more generally applicable to real-world tasks.

HIRO can be roughly introduced by three points (paper 3.1-3.3):

* Hierarchical Frameworks: Low-level policy $\mu^{lo}$ generates action $a$ to reach a goal $g$, it directly interacts with the environment and gets reward $r$, which is used to feed high-level updating. High-level policy $\mu^{hi}$ takes the low-level behavior as observation, then generates goals intermittently (every $c$ steps) for low-level policy to pursue. The high-level policy also generates intrinsic rewards to update low-policy. Low-level policy and high-level policy are both trained with off-policy algorithm TD3, which follows the standard RL setting.
* Parameterized Reward: $\mu^{lo}$ takes $(state, goal)$ as its observation to generate action at each time step. so the off-policy experience tuple for low-level policy is $(s_t, g_t, a_t, r_t, s_{t+1}, g_{t+1})$. Since the task for low-level policy is to reach the goal state generated by high-level policy, i.e., let $s_{t+c} = s_t + g_t$ happen, it takes a dense parameterized reward that represents the L2 distance of current state and goal state, i.e., $r(s_t, a_t, g_t, s_{t+1}) = -||s_t, a_t, g_t, s_{t+1}||_2$. On the other hand, the high-level policy takes the returned provided by environment to encourage it to generate good goal sequences that leads the agent to achieve more rewards. The off-policy experience tuple for low-level policy is $(s_t, g_t, \sum{r_{t:t+c-1}}, s_{t+c})$. 
* Off-Policy Correction: Transitions obtained from past lower-level controllers do not accurately reflect the actions (and therefore resultant states $s_{t+1:t+c}$) that would occur if the same goal were used with the current lower-level controller. HIRO introduces a correction that translates old transitions into ones that agree with the current lower-level controller. The method is to select the goal that most likely to let current low-level policy induce the same low-level behavior with the current instantiation of the lower-level policy. This surrogate goal $\hat{g}$ is generated form a set of candidate goals then replace the original goal in high-level off-policy experience tuple. The paper claims this method works well empirically then also discussed other methods of applying this off-policy correction.

To conclude, form my point of view, HIRO paper majorly contributes to:

* provided an HRL framework that lower-level controllers are supervised with goals that are learned and proposed automatically by the higher-level controllers.
* improve traditional HRL methods at data-efficiency and real-world task generalization ability by utilizing off-policy training algorithm
* proposed off-policy correction method to leverage the low-level/high-level policy incompetence problem introduced by the off-policy algorithm and proves the correction method works well.

### Algorithm

#### **Method Review**

HIRO selects TD3 (paper 2.1) as its off-policy training algorithm. The algorithm for HIRO is basically a combination of TD3 and hierarchical RL framework. 

The low-level and high-level policy is trained with TD3 as usual, but with modified rewards: Low-level policy uses a dense intrinsic reward provided by high-level policy, which is the L2 distance of current state to the goal state. High-level policy uses the cumulated reward from the environment, which is provided by low-level policy. (paper 3.1-3.2)

To apply off-policy correction, HIRO selects an approximately optimal $\hat{g}$ as a surrogate of $g$ to train high-level policy. The applied method is to select the $\hat{g}$ that provides the biggest likelihood to generate past induced low-level behavior. (paper 3.3, C.3, A)

The original paper does not explain its algorithm in pseudo-code. To make the algorithm clear, we compose our own:

#### **Algorithm HIRO**

​		Initialize critic networks $Q_{\theta_1^{lo}}$, $Q_{\theta_2^{lo}}$, $Q_{\theta_2^{hi}}$, $Q_{\theta_2^{hi}}$, actor networks $\mu_{\phi_1^{lo}}$, $\mu_{\phi_2^{lo}}$, $\mu_{\phi_1^{hi}}$, $\mu_{\phi_1^{hi}}$ with random $\theta s$ and $\phi s$ 

​		Initialize target networks $\theta_1^{'lo} \leftarrow \theta_1^{lo}$,  $\theta_2^{'lo} \leftarrow \theta_2^{lo}$,  $\theta_1^{'high} \leftarrow \theta_1^{hi}$,  $\theta_2^{'hi} \leftarrow \theta_2{hi}$, $\phi_1^{'lo} \leftarrow \phi_1^{lo}$, $\ldots$

​		Initialize replay buffer $\beta^{lo}$, $\beta^{hi}$

​		**for** $t = 1$ **to** $T$ **do**

​				select action with explore-noise $a_t \sim \mu(s_t, g_t) + \epsilon, \epsilon \sim N(0, \sigma)$

​				observe reward $r$ and new state $s_{t+1}$, store tuple $(s_t, g_t, a, r, s_{t+1}, g_{t+1})$ in $\beta_{lo}$

​				select next goal via goal transition model with explore-noise  $g_{t+1} \sim h(s_{t}, g_{t}, s_{t+1}) + \epsilon, \epsilon \sim N(0, \sigma)$

​				**if** $t$ mod $c$ **then**

​						generate next  goal via $\mu^{hi}$ with explore-noise  $g_{t+1} \sim \mu(s_{t+1}) + \epsilon, \epsilon \sim N(0, \sigma)$

​						apply off-policy correction $\hat{g} = correction(g)$, store tuple $(s_{t-c+1}, g_{t-c+1}, r_{t-c+1:t}, s_{t+1})$ in $\beta_{hi}$

​				**end if**

​				sample mini-batch of $N$ steps $(s_t, g_t, a, r, s_{t+1}, g_{t+1})$ from $\beta_{lo}$

​				$\hat{a} \leftarrow \mu_{\phi'}(s', g') + \epsilon, \epsilon \in \text{clip}(N(0, \hat{\sigma}), -k, k)$

​				$y^{lo} \leftarrow r + \gamma \text{min}_{i=1,2}Q_{\theta_i'}(s', g', \hat{a})$

​				update critics $\theta_i^{lo} \leftarrow \text{argmin}_{\theta_i^{lo}}N^{-1}\sum{(y^{lo}-Q_{\theta_i^{lo}}(s,a, g)})^2$ 

​				**if** $t$ mod $d$ **then**

​						update $\phi^{lo}$ by the deterministic policy gradient:

​						soft update target network

​				**end if** 

​				**if** $t$ mod $c$ **then**

​						sample mini-batch of $N$ steps $(s_{t-c+1}, g_{t-c+1}, r_{t-c+1:t}, s_{t+1})$ from $\beta_{hi}$

​						$\hat{g'} \leftarrow \mu_{\phi'}(s') + \epsilon, \epsilon \in \text{clip}(N(0, \hat{\sigma}), -k, k)$

​						$y^{hi} \leftarrow r + \gamma \text{min}_{i=1,2}Q_{\theta_i'}(s', \hat{g'})$

​						update critics $\theta_i^{hi} \leftarrow \text{argmin}_{\theta_i^{hi}}N^{-1}\sum{(y^{hi}-Q_{\theta_i^{hi}}(s, g, a)})^2$ 

​						**if** $t$ mod $d$ **then**

​								update $\phi^{hi}$ by the deterministic policy gradient:

​								soft update target network

​						**end if** 

​				**end if**

​		**end for**

where $k$ is explore-noise boundary

## Implementation Structure

The code structure of this implementation is simpler compared with the original implementation. This implementation basically follows the structure of an off-policy training code structure. Below is a quick view of all crucial python files.

*  run.py: take in command from a terminal, parse arguments and launch experiments accordingly
*  hiro.py: contains the training loop as the core code segment. It trains the algorithms and uses components provided by other python files
*  networks.py: holds NNs used in training process
*  utils.py: holds util classes, functions and utility data
*  experience_buffer.py: holds the experience buffer class

<img src=".\readme_data\code_structure.jpg" alt="code_structure" style="zoom:65%;" />

**Hint:**

* only uses positional parameters to form goal
* set delayed parameter update interval to 1

## Start to Use This Code Base

### requirements

Please make sure you have the following software correctly installed before using this codebase:

* PyTorch
* OpenAI Gym (Mujoco)
* Weights & Biases

### Install

#### Clone Repo & Deploy

clone this codebase:

> git clone https://github.com/ziangqin-stu/impl_data-effiient-hrl.git

go to */impl_hirol*, install required packages:

> pip install -e requirements.txt

#### Test Run

Set your wandb, modify *./test/td3.py* line 162 to your project name.

In terminal, go to project root path, type command to run a simple TD3 training process to test success of deploy:

```python
python run.py --algorithm=td3 --param_id=1
```

This experiment runs on CPU, the final results should be like this: 

<div style="display:flex;">
    <div style="display:flex; margin:auto;">
        <img src=".\readme_data\td3-idp-reward.png" alt="td3-idp-reward.png" width="550" style="padding: 5px;"/>
        <img src=".\readme_data\td3-idp.gif" alt="td3-idp" width="250" style="padding: 5px;"/>
    </div>       
</div>

I tested the TD3 algorithm implementation with "InvertedPendulum-v2", "InvertedDoublePendulum-v2", "Hopper-v2", "Ant-v2", "Humanoid-v2", all experiments are successful:

  <div style="display:flex;">
      <div style="display:flex; margin:auto;">
          <img src=".\readme_data\td3-ip.gif" alt="td3-ip" width="200" style="padding: 5px;"/>
          <img src=".\readme_data\td3-hopper.gif" alt="td3-hopper" width="200" style="padding: 5px;"/>
          <img src=".\readme_data\td3-ant.gif" alt="td3-ant" width="200" style="padding: 5px;"/>
          <img src=".\readme_data\td3-humanoid.gif" alt="td3-humanoid" width="200" style="padding: 5px;"/>
      </div> 
  </div>




### Launch Experiments & Play Code

#### Command Line Interface

This codebase allows users to run experiments with accessibility to every parameter from command-line interface. Read below guides to learn how to use this functional:

* Check `train_param_td3.csv` and `train_param_hiro.csv` to select arguments for TD3 or HIRO training 

  * you can check `run.py` line 18-57 to understand these parameters
  * each line of the parameters is a complete experiment setting, remember `param_id` of the setting you select for future use

* Launch an experiment from terminal

  * in a terminal, navigate to project root folder.

  * type `(CUDA_VISIBLE_DEVICE=x) python run.py --alg={td3/hiro} --param_id={y}` to launch a experiment strictly follow the setting store in `.csv` files

  * type more command-line arguments to customize your experiments setting, for example:

    `(CUDA_VISIBLE_DEVICE=1) python run.py --alg=hiro --param_id=1 --start_timestep=250000 --policy_freq=2`

  * note that `td3` algorithm can only run on CPU

* Add your customized experiment settings to `.csv` file to create a convenient baseline for training commands

#### Evaluation

Not implement.

## Experiment Results

### Ant Push

* **Evaluation Success Rate**

  <div style="display:flex;">
      <div style="display:flex; margin:auto;">
          <img src=".\readme_data\hiro-push-success.png" alt="hiro-push-success" width="550" style="padding: 5px;"/>
          <img src=".\readme_data\paper-fig4.png" alt="paper-fig4" width="250" style="padding: 5px;"/>
      </div>
  </div>


  Evaluate policies every 10k steps, each time run 50 episodes with 5 random seed (10 episode each). The success condition is the same with that in HIRO paper. 

  The result shows my experiment reaches a higher success rate (~ 0.8, original result ~ 0.6), but takes more steps to find correct path.

  The experiment is super sensitive to random seed. I select seed to be 0, 123, 1234, 54321, only seed = 0 and seed = 123 works.

* **Segment reward (each 10 time steps) for high-level policy:**

  <div style="display:flex;">
      <div style="display:flex; margin:auto;">
          <img src=".\readme_data\hiro-push-epiR_h.png" alt="hiro-push-epiR_h" width="550" style="padding: 5px;"/>
          <img src=".\readme_data\hiro-push-epiR_h_detail.png" alt="hiro-push-epiR_h_detail" width="250" style="padding: 5px;"/>
      </div>
  </div>

  
  The Y-axis of the above plot is the high-level policy reward for each $c $ steps. The left image shows the detail of this reward curve: In an episode, high-level reward ascends from ~190 (agent at initial state (0, 0), target state (0, 19), L2 distance is ~19, accumulated high-level reward is ~($19*c$)$=190$) to 90 (agent push the movable block forward and be blocked at around (0, 10)), or to less than 50 (agent walk in target room).


* **Segment reward (each $c$=10 time steps) for low-level policy:**

  <div style="display:flex;">
      <div style="display:flex; margin:auto;">
          <img src=".\readme_data\hiro-push-epiR_l.png" alt="hiro-push-epiR_l" width="550" style="padding: 5px;"/>
      </div>
  </div> 

  The Y-axis of the above plot is the low-level policy reward for each $c $ steps. This reward does not ascend since it is depend on how high-level policy selects goal, and this goal selecting process is independent with low-level policy. The learning process of low-level policy is not clearly reflected in this curve. But we can see the agent learned how to move in episodes video.

* **Episodes Video:**


  * Learn how to walk:

      <div style="display:flex;">
          <div style="display:flex; margin:auto;">
              <img src=".\readme_data\hiro-push-init.gif" alt="hiro-push-init" width="350" style="padding: 5px;"/>
              <img src=".\readme_data\hiro-push-learnmove.gif" alt="hiro-push-learnmove" width="350" style="padding: 5px;"/>
          </div> 
      </div>

  * Try different paths:

      <div style="display:flex;">
          <div style="display:flex; margin:auto;">
              <img src=".\readme_data\hiro-push-straight_3.gif" alt="hiro-push-straight" width="350" style="padding: 5px;"/>
              <img src=".\readme_data\hiro-push-right.gif" alt="hiro-push-right" width="350" style="padding: 5px;"/>
          </div> 
      </div>

  * Practice the correct path:

      <div style="display:flex;">
      	<div style="display:flex; margin:auto;">
              <img src=".\readme_data\hiro-push-correcttry.gif" alt="hiro-push-correcttry" width="350" style="padding: 5px;"/>
              <img src=".\readme_data\hiro-push-success_2.gif" alt="hiro-push-success" width="350" style="padding: 5px;"/>
          </div>     
      </div>

### Ant Fall (waiting)

​	No success experiment for now

## Project Summary

#### Things Learned

* Value-Based RL algorithms and its variants
* Time Difference optimization method
* TD3 algorithm implementation
* HRL/HIRO framework and its implementation

#### Implementation Skills

* Iteration in development: start with a minimal working set then extend it or strengthen it
  * helps to quick locate issues
  * helps to shrink down test-experiments' time
* Ask for help timely

#### Shortage:

* Not plan well
  * understanding of paper is not enough before start implement.
* Too ambitious with implementation speed
  * hidden logic flaws
  * wrong algorithm details
  * repletely debug over same issue
* Ask too little
  * waste time and energy
  * waste opportunity of practicing communication and conclude/present problems 