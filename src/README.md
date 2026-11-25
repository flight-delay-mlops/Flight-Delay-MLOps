# The model

How the flight delay model works, and the two ways it can deal with the rare delay classes:
**weighted gradients** (the default) and **decision weights**.

| File | What it holds |
|---|---|
| [prepare.py](prepare.py) | Cleans and labels the flights, splits them by date |
| [features.py](features.py) | Delay rates of the 30 days before each flight |
| [fdt.py](fdt.py), [forest.py](forest.py) | The soft oblique tree, and the forest of them |
| [models.py](models.py) | Every model `train.model` can choose, and the decision weights |
| [train.py](train.py) | Fits, tunes (only decision weights), evaluates and logs the model |
| [promote.py](promote.py) | Moves `@production` to the `@staging` model |

## What goes in

All known before departure:

- **Categories**, one-hot encoded: airline, origin, destination, departure hour block.
- **Numbers**, with missing ones replaced by the average, then scaled: distance, and six delay
  rates (average delay and share of flights 15+ minutes late, over the 30 days before the
  flight's date, for its airport, airline and route).

The encoding is part of the saved model, so the API applies exactly the same one.

Both sides use the same [Feast](https://feast.dev) definitions (`api/features.py`). Training
gets the delay rates from its historical features and the API from its online store
([api/README.md](../api/README.md#feature-store)).

## The forest

`decision_forest` is five soft, oblique decision trees of depth 5 (`decision_tree` is one of
depth 6).

- **Oblique**: each node looks at a weighted sum of all the inputs, not at a single one.
- **Soft**: a node doesn't send a flight left or right. It sends it left with probability
  `sigmoid(x · W + b)` and right with the rest, so every flight reaches every leaf.

For a flight `x`, with leaves `l` and classes `c`:

```
P(leaf l | x)  = product, over the nodes on the way to l, of the probability of that turn
P(class c | l) = softmax(theta_l)
tree(x)        = sum over l of P(leaf l | x) · P(class c | l)
forest(x)      = average of the five trees
```

The probabilities of a tree's leaves add up to 1, and so do the class probabilities of a leaf.
That makes the output of a tree, and of the forest, real probabilities. In the code each tree
returns `log(probability)`: summing logs is more stable than multiplying many small numbers, and
a softmax over them gives back the same probabilities, so the usual cross-entropy is correct.

Training minimises that cross-entropy with Adam (3 epochs, batches of 1024). All five trees are
trained together on their average; they differ only in their random starting point.

## The rare classes

In the training flights, 78.6% are on time, 14.3% moderately late and 7.2% severely late. A
model that always says "on time" has 81% accuracy on validation, but a macro-F1 (the average F1
of the three classes) of only 0.298. `train.imbalance` chooses the correction:

| `train.imbalance` | Where the weights act | `predict_proba` |
|---|---|---|
| `gradient_weights` | In the loss, while training | Shifted towards the rare classes |
| `decision_weights` | After training, when choosing the label | The real probabilities |

Only one is used at a time: both together would correct twice. The logistic regression always
uses balanced class weights, and the baselines use none.

## Weighted gradients

Each flight's loss, and so its gradient, is multiplied by the weight of its class. Errors on rare
classes then move the trees more:

```
weight_c = (flights / flights of class c) ^ class_weight_power
loss     = sum of weight · cross-entropy / sum of weight     (per batch)
```

Power 0 means no weights and 1 makes the three classes count the same. We compared powers on
200,000 flights of April to June 2023, with the model fitted on 1,000,000 flights before them
(weights relative to on_time):

| Power | Moderate | Severe | Accuracy | Macro-F1 |
|---|---|---|---|---|
| 0.5 | 2.35 | 3.35 | 0.737 | 0.404 |
| **0.6** | 2.79 | 4.26 | **0.712** | **0.422** |
| 0.7 | 3.32 | 5.43 | 0.677 | 0.431 |
| 0.75 | 3.61 | 6.13 | 0.655 | 0.430 |
| 0.8 | 3.93 | 6.91 | 0.629 | 0.426 |
| 0.9 | 4.67 | 8.80 | 0.568 | 0.409 |
| 1.0 | 5.54 | 11.21 | 0.492 | 0.378 |

Full balancing (1.0) is clearly too much. We use 0.6, the best macro-F1 + 0.5 × accuracy, because
both metrics are evaluated. 0.7 has the best macro-F1 alone.

No months are kept for tuning, so the final model is also fitted on April to June 2023. The
probabilities are those of the weighted problem, so don't read them as the chance of a delay.

## Decision weights

With `decision_weights` the trees train unweighted and learn the real probabilities.

Severe delays are rarely the *most likely* outcome, so picking the highest probability almost
never predicts them. Instead, we multiply each probability by a weight and pick the highest:

```
label = argmax over c of  weight_c · P(c | x)
```

Example, with the weights of the run below (on time 1, moderate 3.47, severe 5.50):

| | on_time | moderate_delay | severe_delay | Label |
|---|---|---|---|---|
| Probability | 0.60 | 0.20 | 0.20 | on_time |
| Probability × weight | 0.60 | 0.69 | **1.10** | severe_delay |

### How they are tuned

```
2022-01 ────────── 2023-03 │ 2023-04 ── 2023-06 │ 2023-07 ── 2023-12
   fit the model           │ tune the weights   │ validation
   (train.sample_rows)     │ (train.tune_rows)  │ (quality gate)
```

1. Fit the model on a random sample of the flights before `train.tune_start`.
2. Predict the probabilities of flights **after** them, which the model has never seen.
3. Fix on_time's weight at 1 (only the ratios matter) and try 41 values between 0.5 and 20 for
   each of the other two: 1681 combinations.
4. Keep the one with the best macro-F1, unless plain probabilities do better.
5. Evaluate once on the validation set, which had no part in choosing them.

Tuning on later flights imitates production, where the model is always used on the future. Using
the validation set instead would make its score, and so the quality gate, too optimistic. The
price is that the model isn't fitted on the last three months of the training data.

The weights are found in `WeightedDecision` ([models.py](models.py)), which wraps the fitted
model. They are saved with it in MLflow, so the API needs no change. With this option every
model is tuned except the baselines (`most_frequent`, `stratified`).

**Labels only.** `predict` uses the weights; `predict_proba` still returns the model's own
probabilities.

## Results

`decision_forest`, 1,000,000 training flights, seed 42, on the 3,463,444 validation flights:

| | Full balancing (first version) | Decision weights | **Gradients, 0.6 (default)** | Gradients, 0.7 |
|---|---|---|---|---|
| Accuracy | 0.540 | 0.726 | **0.762** | 0.728 |
| Macro-F1 | 0.377 | 0.424 | **0.416** | 0.424 |
| F1 on_time | 0.700 | 0.848 | 0.871 | 0.850 |
| F1 moderate_delay | 0.235 | 0.219 | 0.172 | 0.203 |
| F1 severe_delay | 0.195 | 0.206 | 0.206 | 0.219 |

The first version also had leaf scores that weren't probabilities. The decision weights were
tuned for macro-F1 alone. The gradients models were fitted on three more months. The power was
chosen on April to June, not on validation.

## Limits

- The decision weights fit the mix of delays in the tuning months, and the class weights that of
  the training flights. If it changes (e.g. a bad winter), they go out of date: train again.
- The delay rates come from recent labels. While the API's feature store is empty, the model
  falls back to the average rates, so it can do worse than in validation.
- One run with one seed: small differences between models may be noise.

## Running it

```bash
python -m src.train    # only the training, with the model in params.yaml
dvc repro              # every stage whose inputs changed, and updates dvc.lock
```

MLflow logs, for each run, the metrics above, and with decision weights also
`macro_f1_untuned` (without them) and `decision_weight_<class>`.
