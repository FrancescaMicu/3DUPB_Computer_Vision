# Procedural Level Generation & Difficulty Classification with SmolLM2, Genetic Algorithms, Q-Learning & ML

Project developed for the **3D UPB Summer School 2026** (Computer Vision Workshop).

---

## Overview

This project implements an end-to-end procedural content generation (PCG) pipeline for **TreeClimb**, a grid-based puzzle-platformer game. The system combines generative AI, evolutionary optimization, reinforcement learning, and supervised machine learning to create playable levels, test them autonomously, and classify their true gameplay difficulty.

Instead of relying on arbitrary difficulty tags, the pipeline closes the loop: **the LLM dreams the level, the Genetic Algorithm builds it properly, Q-Learning plays it, and Logistic Regression gives the final verdict.**

---

## Game Mechanics & Tiles

The game is played on a **16x16 grid** where the agent starts at row 15 (`S`) and must reach the exit at row 0 (`E`) while collecting all Golden Ladders (`+`) as mandatory checkpoints:

| Tile | Symbol | Description |
| :---: | :---: | :--- |
| **Bird** | `.` | Solid wall / blocking obstacle |
| **Normal Ladder** | `N` | Walkable path |
| **Start** | `S` | Agent spawn point (fixed at bottom row) |
| **Exit** | `E` | Goal tile (fixed at top row, minimum 5 columns offset from `S`) |
| **Beetle** | `K` | Enemy penalty tile |
| **Portal** | `P` | Bidirectional teleport pairs |
| **Vines** | `V` | Boost mechanic (skips 1 cell upward) |
| **Broken Ladder** | `B` | Trap mechanic (forces downward slide) |
| **Golden Ladder** | `+` | Mandatory checkpoints (strictly 2 per level) |

---

## Architecture Pipeline

1. **LLM Level Generation (`SmolLM2-1.7B-Instruct`)**
   - Generates initial 16x16 ASCII layouts using structured prompts and temperature sampling.
   - Run through a rule-based repair script (`fix_grid`) to link ladders and enforce structural requirements.

2. **Validation & Repair (BFS Pathfinding)**
   - Custom Breadth-First Search ensures topological solvability (`Start` $\to$ `Checkpoint 1` $\to$ `Checkpoint 2` $\to$ `Exit`) while handling portals, vine jumps, and broken ladder slides.
   - Isolated islands and unreachable hazards are purged.

3. **Difficulty Shaping (Genetic Algorithm)**
   - Population of maps evolved through mutation, tournament selection, and crossover.
   - Fitness functions fine-tune level difficulty (`easy`, `medium`, `hard`) by weighting path length, enemy density, traps, and movement mechanics.

4. **Automated Playtesting (Q-Learning Agent)**
   - An RL agent explores each map across thousands of episodes.
   - Extracts realistic gameplay metrics: steps taken, cumulative reward, and late-training win rate (`succ[-100:]`).
   - Generates greedy rollouts and policy visualization maps.

5. **Empirical Difficulty Classification (Logistic Regression)**
   - Supervised classification using structural map features along with agent behavioral telemetry.
   - Evaluated via `StratifiedKFold(n_splits=3, shuffle=True)` with balanced class weighting.
   - Achieves consistent $\sim 99\%$ multi-class accuracy across Easy, Medium, and Hard tiers.

---

## Dataset & Model Features

The classification dataset contains 113 unique levels characterized by:
- Map elements: `normal_ladders`, `birds`, `beetles`, `vines`, `broken_ladders`
- Agent telemetry: `len` (unique visited tiles), `steps`, `reward`
- Outcome metrics: `won_greedy` (binary victory) and `win_rate` (continuous performance metric over last 100 episodes)
