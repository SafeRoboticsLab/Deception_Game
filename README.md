# Deception game: Closing the safety-learning loop in interactive robot autonomy

[![License][license-shield]][license-url]
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue)](https://www.python.org/downloads/)


[Haimin Hu](https://haiminhu.org/)<sup>1</sup>,
[Zixu Zhang](https://zzx9636.github.io/)<sup>1</sup>,
[Kensuke Nakamura](https://kensukenk.github.io/),
[Andrea Bajcsy](https://www.cs.cmu.edu/~abajcsy/),
[Jaime F. Fisac](https://saferobotics.princeton.edu/jaime)

<sup>1</sup>equal contribution

Published as a conference paper at CoRL'2023.


<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/SafeRoboticsLab/Deception_Game">
    <img src="misc/road_crossing.gif" alt="Logo" width="800">
  </a>
  <p align="center">
  </p>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#installation">Installation</a></li>
    <li><a href="#training">Training</a></li>
    <li><a href="#closed-loop-simulation">Closed-loop Simulation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This repository implements a general RL-based framework for approximate HJI Reachability analysis in joint physical-belief spaces.
The control policies explicitly account for a robot's ability to learn and adapt at runtime.
The repository is primarily developed and maintained by [Haimin Hu](https://haiminhu.org/) and [Zixu Zhang](https://zzx9636.github.io/).

Click to watch our spotlight video:
[![Watch the video](misc/Deception_Game.jpg)](https://haiminhu.org/wp-content/uploads/2024/06/deception_game.mp4)


## Installation
This repository relies on [`ISAACS`](https://github.com/SafeRoboticsLab/ISAACS). Please follow the instructions there to set up the environment.


## Training
Please follow these steps to train the control and disturbance/adversary policies.
+ Pretrain a control policy
  ```bash
  python script/bgame_intent_pretrain_ctrl.py
  ```
+ Pretrain a disturbance policy
  ```bash
  python script/bgame_intent_pretrain_dstb.py
  ```
+ Joint control-disturbance training
  ```bash
  python script/bgame_intent_isaacs.py
  ```
To train the baseline policies, replace `bgame` with `robust` and repeat the above steps.


## Closed-loop Simulation
We provide a [Notebook](https://github.com/SafeRoboticsLab/Deception_Game/blob/main/simulation.ipynb) for testing the trained policies in closed-loop simulations and comparing with baselines.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/SafeRoboticsLab/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/SafeRoboticsLab/SHARP/contributors
[forks-shield]: https://img.shields.io/github/forks/SafeRoboticsLab/repo.svg?style=for-the-badge
[forks-url]: https://github.com/SafeRoboticsLab/SHARP/network/members
[stars-shield]: https://img.shields.io/github/stars/SafeRoboticsLab/repo.svg?style=for-the-badge
[stars-url]: https://github.com/SafeRoboticsLab/SHARP/stargazers
[issues-shield]: https://img.shields.io/github/issues/SafeRoboticsLab/repo.svg?style=for-the-badge
[issues-url]: https://github.com/SafeRoboticsLab/SHARP/issues
[license-shield]: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
[license-url]: https://opensource.org/licenses/BSD-3-Clause
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/SafeRoboticsLab
[homepage-shield]: https://img.shields.io/badge/-Colab%20Notebook-orange
[homepage-url]: https://colab.research.google.com/drive/1_3HgZx7LTBw69xH61Us70xI8HISUeFA7?usp=sharing
