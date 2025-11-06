# SeedGraph Prompt Engineering Guide

## 🌍 Controlling Language and Domain

SeedGraph uses Qwen-2.5-0.5B, which is **multilingual** and **domain-flexible**. The language and content type are controlled by your **seed prompt**, not CLI arguments.

## 🇫🇷 French Language Examples

### Example 1: French Sentences
```bash
seedgraph grow \
  --prompt "Les réseaux de neurones apprennent des représentations" \
  --max-nodes 100 \
  --max-depth 4
```

**What happens:**
- Model continues in French (follows the prompt pattern)
- Generates French text branches
- Explores French semantic space
- Example nodes:
  ```
  "Les réseaux de neurones apprennent des représentations"
  → "Les réseaux de neurones apprennent des représentations à partir"
  → "Les réseaux de neurones apprennent des représentations à partir de données"
  → "Les réseaux de neurones apprennent des représentations hiérarchiques"
  ```

### Example 2: French Technical Content
```bash
seedgraph grow \
  --prompt "L'intelligence artificielle utilise des algorithmes d'apprentissage" \
  --max-nodes 500 \
  --max-depth 5
```

### Example 3: French Conversational
```bash
seedgraph grow \
  --prompt "Bonjour, comment allez-vous aujourd'hui?" \
  --max-nodes 200 \
  --max-depth 4
```

## 🐍 Python Code Examples

### Example 1: Python Code Generation
```bash
seedgraph grow \
  --prompt "def process_data(input_list):" \
  --max-nodes 100 \
  --max-depth 5 \
  --top-k 5
```

**Expected output:**
```python
"def process_data(input_list):"
→ "def process_data(input_list): return"
→ "def process_data(input_list): return [x"
→ "def process_data(input_list): for item in"
→ "def process_data(input_list): if not input_list"
```

### Example 2: Python Class Definitions
```bash
seedgraph grow \
  --prompt "class NeuralNetwork:" \
  --max-nodes 200 \
  --max-depth 6
```

### Example 3: Python Docstrings
```bash
seedgraph grow \
  --prompt '"""Calculate the mean of a list of numbers.' \
  --max-nodes 150 \
  --max-depth 4
```

### Example 4: Complete Python Patterns
```bash
seedgraph grow \
  --prompt "import torch\nimport numpy as np\n\ndef train_model(data, labels):" \
  --max-nodes 300 \
  --max-depth 7
```

## 🌐 Other Languages

### Spanish
```bash
seedgraph grow \
  --prompt "Las redes neuronales aprenden patrones de los datos" \
  --max-nodes 100
```

### German
```bash
seedgraph grow \
  --prompt "Neuronale Netze lernen Muster aus Daten" \
  --max-nodes 100
```

### Chinese
```bash
seedgraph grow \
  --prompt "神经网络从数据中学习模式" \
  --max-nodes 100
```

### Japanese
```bash
seedgraph grow \
  --prompt "ニューラルネットワークはデータからパターンを学習する" \
  --max-nodes 100
```

## 💻 Other Programming Languages

### JavaScript
```bash
seedgraph grow \
  --prompt "function processData(inputArray) {" \
  --max-nodes 100 \
  --max-depth 5
```

### Java
```bash
seedgraph grow \
  --prompt "public class DataProcessor {" \
  --max-nodes 150 \
  --max-depth 5
```

### Rust
```bash
seedgraph grow \
  --prompt "fn process_data(input: Vec<i32>) ->" \
  --max-nodes 100 \
  --max-depth 5
```

### SQL
```bash
seedgraph grow \
  --prompt "SELECT users.name, orders.total FROM users JOIN" \
  --max-nodes 100 \
  --max-depth 4
```

## 📋 Domain-Specific Examples

### Medical/Healthcare
```bash
seedgraph grow \
  --prompt "The patient presents with symptoms of acute" \
  --max-nodes 200 \
  --max-depth 5
```

### Legal
```bash
seedgraph grow \
  --prompt "In accordance with Section 42 of the statute, the defendant" \
  --max-nodes 200 \
  --max-depth 5
```

### Scientific Paper Style
```bash
seedgraph grow \
  --prompt "We propose a novel approach to deep learning that utilizes" \
  --max-nodes 300 \
  --max-depth 6
```

### Business/Finance
```bash
seedgraph grow \
  --prompt "The quarterly revenue increased by 15% due to" \
  --max-nodes 200 \
  --max-depth 5
```

## 🎯 Tips for Effective Prompting

### 1. **Be Specific and Consistent**
❌ Bad: `"code"`
✅ Good: `"def calculate_statistics(data):"`

The model follows your pattern!

### 2. **Use Natural Starting Points**
```bash
# For French technical content
--prompt "Les algorithmes de machine learning peuvent"

# For Python functions
--prompt "def compute_metrics(predictions, labels):"

# For English narrative
--prompt "Once upon a time, in a distant galaxy,"
```

### 3. **Multi-line Prompts for Code**
```bash
# Use literal strings for multi-line
seedgraph grow --prompt "import pandas as pd
import numpy as np

def analyze_data(df):"
```

### 4. **Domain Priming**
Include domain-specific terms in your prompt:

```bash
# Medical
--prompt "The patient's electrocardiogram showed signs of"

# Legal
--prompt "Pursuant to clause 3.2 of the agreement,"

# Math
--prompt "Let f(x) be a continuous function such that"
```

## 🔬 Advanced: Mixed Content

### Code with Comments
```bash
seedgraph grow \
  --prompt "# Calculate fibonacci numbers
def fibonacci(n):" \
  --max-nodes 200
```

### Bilingual (Code + Docstring)
```bash
seedgraph grow \
  --prompt "def traiter_donnees(liste):
    \"\"\"Traite une liste de données en français." \
  --max-nodes 150
```

### Technical Writing
```bash
seedgraph grow \
  --prompt "Algorithm 1: Gradient Descent
1. Initialize weights randomly
2. For each iteration:" \
  --max-nodes 200
```

## ⚠️ Important Notes

### Language Capability
Qwen-2.5-0.5B supports:
- ✅ English (excellent)
- ✅ Chinese (excellent)
- ✅ French (good)
- ✅ Spanish (good)
- ✅ German (good)
- ✅ Code (all major languages)
- ⚠️ Other languages (variable quality)

### Model Behavior
1. **Follows prompt pattern**: If you start in French, it continues in French
2. **Can mix languages**: If prompt has code + comments, it maintains that
3. **Context-aware**: Technical prompts → technical continuations
4. **Not perfect**: May occasionally switch languages (rare)

### Quality Tips
```bash
# For better language consistency, use longer prompts
--prompt "Les réseaux de neurones profonds utilisent des couches multiples pour apprendre"
# (More context = more consistent language)

# For code, include typical syntax
--prompt "class DataProcessor:
    def __init__(self, config):"
# (Structural clues help maintain format)
```

## 📊 Practical Examples

### 1. French NLP Research
```bash
seedgraph grow \
  --prompt "Le traitement automatique du langage naturel en français utilise des modèles de transformers qui" \
  --max-nodes 500 \
  --max-depth 6 \
  --run-id french_nlp
```

### 2. Python ML Pipeline
```bash
seedgraph grow \
  --prompt "import sklearn
from sklearn.pipeline import Pipeline

def create_ml_pipeline(steps):" \
  --max-nodes 300 \
  --max-depth 7 \
  --run-id python_ml
```

### 3. Multilingual Comparison
```bash
# English version
seedgraph grow \
  --prompt "Machine learning algorithms learn patterns from data" \
  --max-nodes 500 \
  --run-id ml_english

# French version (compare semantic spaces)
seedgraph grow \
  --prompt "Les algorithmes d'apprentissage automatique apprennent des motifs à partir de données" \
  --max-nodes 500 \
  --run-id ml_french

# Compare the resulting graphs!
```

### 4. Code Documentation Generation
```bash
seedgraph grow \
  --prompt "def process_user_input(input_string):
    \"\"\"
    Process and validate user input string.

    Args:" \
  --max-nodes 200 \
  --max-depth 5 \
  --run-id code_docs
```

## 🎨 Creative Applications

### Story Generation (Different Languages)
```bash
# English fantasy
seedgraph grow \
  --prompt "In the ancient kingdom of Eldoria, a young wizard discovered" \
  --max-nodes 300

# French mystery
seedgraph grow \
  --prompt "Dans les ruelles sombres de Paris, le détective découvrit" \
  --max-nodes 300
```

### Poetry Patterns
```bash
# English haiku-like
seedgraph grow \
  --prompt "Silent moonlight falls
Upon the ancient temple" \
  --max-nodes 100

# French alexandrine
seedgraph grow \
  --prompt "Sous le ciel étoilé de la nuit profonde" \
  --max-nodes 100
```

## 🚀 Workflow: Generate Domain-Specific Graphs

```bash
# Step 1: Small test to verify language/domain
seedgraph grow \
  --prompt "Votre texte français ici" \
  --max-nodes 50 \
  --no-pca

# Step 2: Check first few nodes to verify pattern
# (Look at checkpoint or verbose output)

# Step 3: Scale up if pattern is correct
seedgraph grow \
  --prompt "Votre texte français ici" \
  --max-nodes 1000 \
  --max-depth 6

# Step 4: Analyze resulting semantic space
```

## 💡 Summary

**No CLI flags needed!** Just:
1. Write your prompt in the target language/domain
2. Model follows the pattern
3. Graph explores that specific semantic space

**The prompt IS the control mechanism.**
