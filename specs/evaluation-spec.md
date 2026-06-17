# Evaluation Spec — Pod Classifier

Complete this spec **before** writing any code for Milestone 3.

Use Plan or Ask mode to think through each blank field. When you're done,
your answers here become the blueprint for `compute_accuracy()` and
`compute_per_class_accuracy()` in `evaluate.py`.

---

## Background: What is evaluation?

After building a classifier, we need to know how well it works. Evaluation answers:
- **Overall:** What fraction of episodes did we classify correctly?
- **Per-class:** Are we better at some labels than others?

Both functions take the same inputs: a list of predicted labels and a list of
ground-truth labels, in the same order.

---

## compute_accuracy(predictions, ground_truth)

### What it does
Returns the fraction of predictions that exactly match the ground truth.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`, one per episode. |
| `ground_truth` | `list[str]` | The correct labels, in the same order as `predictions`. |

### Output

| Return value | Type | Description |
|---|---|---|
| accuracy | `float` | A value between 0.0 and 1.0. |

---

### Spec fields — fill these in before writing code

**Formula:**

```
Accuracy is the number of positions where the predicted label exactly matches the ground-truth label, divided by the total number of ground-truth labels.
```

---

**Step-by-step logic:**

```
1. If the ground-truth list is empty, return 0.0.
2. Compare each predicted label with the ground-truth label at the same position.
3. Count how many pairs match exactly.
4. Divide the number of correct matches by the total number of ground-truth labels.
5. Return the result as a float between 0.0 and 1.0.
```

---

**Edge case — what if both lists are empty?**

```
Return 0.0 because there are no examples to evaluate, and this avoids division by zero.
```

---

**Worked example:**

```
predictions  = ["interview", "solo", "panel", "interview"]
ground_truth = ["interview", "solo", "solo",  "narrative"]

Two predictions are correct: "interview" and "solo".
Accuracy = 2 correct / 4 total = 0.5.
Therefore, compute_accuracy() returns 0.5.
```

---

## compute_per_class_accuracy(predictions, ground_truth)

### What it does
Returns accuracy broken down by each label. For each label in `VALID_LABELS`,
reports how many episodes with that ground-truth label were classified correctly.

### Inputs

| Parameter | Type | Description |
|---|---|---|
| `predictions` | `list[str]` | Labels predicted by `classify_episode()`. |
| `ground_truth` | `list[str]` | Correct labels, in the same order. |

### Output

A `dict` keyed by label. Each value is a dict with three keys:

```python
{
    "interview": {"correct": int, "total": int, "accuracy": float},
    "solo":      {"correct": int, "total": int, "accuracy": float},
    "panel":     {"correct": int, "total": int, "accuracy": float},
    "narrative": {"correct": int, "total": int, "accuracy": float},
}
```

---

### Spec fields — fill these in before writing code

**What does "correct" mean for a given class?**

```
An episode counts as correct for a class when its ground-truth label is that class and the predicted label exactly matches it. For example, an episode is correct for "interview" only when ground_truth is "interview" and prediction is also "interview".
```

---

**What does "total" mean for a given class?**

```
Total is the number of episodes whose ground-truth label is that specific class. It is not the total number of predictions across every class.
```

---

**Step-by-step logic:**

```
1. Initialize a result entry for every label in VALID_LABELS with correct = 0, total = 0, and accuracy = 0.0.
2. Loop over each predicted label and ground-truth label together.
3. For each pair, use the ground-truth label to find the matching class entry and increase its total by 1.
4. If the predicted label exactly matches the ground-truth label, increase that class's correct count by 1.
5. After the loop, calculate accuracy for each class as correct / total when total is greater than 0; otherwise leave accuracy as 0.0.
6. Return the completed dictionary.
```

---

**Edge case — what if a class has no examples in ground_truth (total == 0)?**

```
Set that class's accuracy to 0.0 because there are no examples for that class, and this avoids division by zero.
```

---

**Worked example:**

```
predictions  = ["interview", "interview", "solo", "panel", "panel"]
ground_truth = ["interview", "solo",      "solo", "panel", "narrative"]

label       correct  total  accuracy
----------  -------  -----  --------
interview   1        1      1.0
solo        1        2      0.5
panel       1        1      1.0
narrative   0        1      0.0
```

---

## Reflection questions (discuss at the checkpoint)

1. Your overall accuracy might be decent even if one class has very low accuracy.
   Why is per-class accuracy a more informative metric than overall accuracy alone?

2. If `panel` episodes consistently get misclassified as `interview`, what does
   that tell you about your training labels or your prompt?

3. You labeled 20 training episodes and evaluated on 20 test episodes (5 per class).
   How might the evaluation results change if you had labeled 100 training episodes?
   What if you had 200 test episodes?
