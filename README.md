# Intelligent Hangman Solver

An advanced Hangman game solver that combines **Hidden Markov Models (HMM)** and **Reinforcement Learning (Q-Learning)** to achieve high success rates in word guessing.

## 🎯 Overview

This project implements an AI agent that learns to play Hangman optimally by:
- **Learning letter patterns** using Hidden Markov Models (position-based, bigrams, trigrams)
- **Making strategic decisions** through Q-Learning reinforcement learning
- **Adapting gameplay** based on revealed letters and remaining lives

## 📁 Project Structure

```
jk/
├── 50k_hmm_rl.py           # Python script version
├── 50k_hmm_rl.ipynb        # Jupyter notebook version
├── corpus.txt              # Training corpus (word list)
├── test.txt                # Test set for evaluation
├── hmm_model.pkl           # Saved HMM model (generated)
├── q_table.json            # Saved Q-table (generated)
├── evaluation_results.json # Evaluation metrics (generated)
└── README.md               # This file
```

## 🚀 Features

### Hidden Markov Model (HMM)
- **Position-based frequency analysis**: Learns which letters appear at specific positions
- **Bigram patterns**: Analyzes two-letter combinations for context
- **Trigram patterns**: Uses three-letter patterns for advanced prediction
- **Length-specific models**: Separate models for each word length
- **Adaptive weighting**: Adjusts strategy based on game progress

### Reinforcement Learning Agent
- **Q-Learning algorithm**: Learns optimal action-value function
- **Epsilon-greedy exploration**: Balances exploration vs exploitation
- **Dynamic epsilon decay**: Gradually shifts from exploration to exploitation
- **Reward shaping**: Sophisticated reward system that:
  - Rewards correct guesses (scaled by letters revealed)
  - Provides life-based bonuses
  - Penalizes wrong guesses (increasing penalty as lives decrease)
  - Gives substantial win/loss bonuses

### Intelligent Decision Making
- **Hybrid scoring**: Combines Q-values with HMM predictions
- **Context-aware**: Uses surrounding letters to make better guesses
- **Vowel prioritization**: Early game vowel bonus
- **Progress tracking**: Adapts strategy based on revealed letters

## 📋 Requirements

```bash
pip install numpy
```

### Optional (for Jupyter notebook)
```bash
pip install jupyter
```

## 🔧 Installation

1. **Clone or download** this repository
2. **Ensure you have** `corpus.txt` and `test.txt` in the same directory
3. **Install dependencies**:
   ```bash
   pip install numpy
   ```

## 💻 Usage

### Option 1: Python Script

Run the complete pipeline:
```bash
python 50k_hmm_rl.py
```

This will:
1. Load the training corpus
2. Train the HMM model
3. Train the RL agent (5000 episodes)
4. Evaluate on the test set
5. Save models and results

### Option 2: Jupyter Notebook

For interactive exploration:
```bash
jupyter notebook 50k_hmm_rl.ipynb
```

Run cells sequentially to:
- Train models step by step
- Visualize training progress
- Test on individual words
- Load pre-trained models
- Experiment with parameters

## 📊 Performance Metrics

The evaluation provides comprehensive metrics:

- **Success Rate**: Percentage of games won
- **Average Wrong Guesses**: Mean number of incorrect guesses per game
- **Total Wrong Guesses**: Sum of all wrong guesses
- **Total Repeated Guesses**: Count of repeated letter selections
- **Final Score**: Calculated as: `(Success Rate × 2000) - (Total Wrong × 5) - (Total Repeated × 2)`

## 🎮 How It Works

### Training Phase

1. **HMM Training**:
   - Groups words by length
   - Builds position-based frequency tables
   - Analyzes bigram and trigram patterns
   - Creates separate models for each word length

2. **RL Training**:
   - Plays 5000 training episodes
   - Uses epsilon-greedy exploration
   - Updates Q-values using temporal difference learning
   - Gradually reduces exploration rate

### Playing Phase

For each guess:
1. **Get HMM scores** for all available letters
2. **Retrieve Q-values** for the current state
3. **Combine scores**: Q-value + (HMM score × 0.7)
4. **Select letter** with highest combined score
5. **Update game state** and continue

## 🔍 Model Details

### HMM Structure
```python
model = {
    'position_freq': {position: {letter: count}},
    'letter_freq': {letter: count},
    'bigram_freq': {bigram: count},
    'trigram_freq': {trigram: count},
    'total_words': int
}
```

### RL Hyperparameters
- **Learning rate (α)**: 0.2
- **Discount factor (γ)**: 0.98
- **Initial epsilon (ε)**: 1.0
- **Epsilon decay**: 0.9985
- **Minimum epsilon**: 0.05
- **Training episodes**: 5000

### State Representation
`state = f"{masked_word}:{sorted_guessed_letters}:{lives_left}"`

Example: `"_a_a_:aent:4"`

## 📈 Training Tips

1. **More episodes = better performance**: Increase episodes for complex corpora
2. **Adjust epsilon decay**: Slower decay allows more exploration
3. **Tune reward values**: Experiment with reward/penalty magnitudes
4. **Quality corpus**: Use diverse, representative training words

## 🎯 Example Output

```
============================================================
EVALUATION RESULTS
============================================================
Total Games:           1000
Games Won:             850
Games Lost:            150
Success Rate:          85.00%
Avg Wrong Guesses:     2.15
Total Wrong Guesses:   2150
Total Repeated:        5

FINAL SCORE:           1279.50
============================================================
```

## 🛠️ Customization

### Modify Training Parameters

```python
agent = RLAgent(
    hmm,
    alpha=0.2,           # Learning rate
    gamma=0.98,          # Discount factor
    epsilon=1.0,         # Initial exploration
    epsilon_decay=0.9985,# Decay rate
    epsilon_min=0.05     # Minimum exploration
)

# Train with custom episodes
agent.train(corpus_words, episodes=10000, verbose=True)
```

### Adjust Scoring Weights

In `HiddenMarkovModel.predict()`:
- Position weight: Line with `* 3.0`
- Bigram weight: Lines with `* 5.0`
- Trigram weight: Lines with `* 8.0`
- Frequency weight: `freq_weight` calculation

In `RLAgent.choose_action()`:
- HMM weight in combined score: `* 0.7`

## 📝 File Formats

### corpus.txt & test.txt
Simple text files with one word per line:
```
apple
banana
cherry
...
```

### hmm_model.pkl
Binary pickle file containing the trained HMM models dictionary.

### q_table.json
JSON file with Q-values:
```json
{
  "state_key": {
    "a": 0.5,
    "b": -0.2,
    ...
  }
}
```

## 🐛 Troubleshooting

**Issue**: `FileNotFoundError: corpus.txt`
- **Solution**: Ensure `corpus.txt` is in the same directory as the script

**Issue**: Training is slow
- **Solution**: Reduce episodes or use a smaller corpus for testing

**Issue**: Low success rate
- **Solution**: Train for more episodes or use a larger/better corpus

**Issue**: Module not found
- **Solution**: Install required packages: `pip install numpy`

## 🤝 Contributing

Feel free to:
- Experiment with different RL algorithms (SARSA, DQN)
- Add more sophisticated NLP features
- Implement neural network approaches
- Optimize hyperparameters
- Improve reward shaping

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Created as an AI-powered Hangman solver combining classical machine learning (HMM) with modern reinforcement learning techniques.

## 🙏 Acknowledgments

- Hidden Markov Models for sequence learning
- Q-Learning for optimal policy discovery
- Epsilon-greedy strategy for exploration-exploitation balance

---

**Note**: Performance depends heavily on the quality and size of your training corpus. For best results, use a diverse corpus representative of your test set.
