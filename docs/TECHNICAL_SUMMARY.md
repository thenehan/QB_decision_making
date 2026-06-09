# Technical Summary

## Methods

The project builds receiver-frame opportunities from NFL Big Data Bowl tracking data. For each dropback, eligible route runners are scored on every frame from ball snap through the throw or play-ending event. The feature set includes receiver movement, quarterback location, throw distance, projected catch point, nearest defenders, throwing-lane defenders, catch-point spacing, and pressure fields such as nearest rusher distance, time to pressure, pressure score, and whether the quarterback is under pressure.

The main model is an XGBoost tree model. It predicts completion probability, interception probability, air yards, YAC, and expected yards. Rare positive labels are class-weighted for the classifier, which mainly helps the interception model because interceptions are rare. Completion and interception probabilities are then calibrated with a light depth/pressure bucket blend before expected yards are calculated. Uncertainty is represented with P25, P50, and P75 expected-yards estimates.

The graph model is a GATv2/LSTM/PNA-style temporal model. It uses player tracking sequences, target-player identity, route one-hot features, and play context features. The GATv2 layers process player-to-player attention on each frame, then an LSTM reads the target receiver's frame sequence in order. A two-view symmetry pass averages the original frame embedding with a 180-degree rotated feature view. Final-frame node embeddings are summarized with PNA-style mean, max, min, and standard-deviation pooling, which is especially useful for YAC. The node feature set now includes QB-facing-target angle, target depth, target distance, nearest rusher distance, pressure score, throwing-lane defender count, closest throwing-lane defender distance, and catch-point nearest defender distance. The graph code can train one full model or separate task models for outcome, air yards, and YAC. It saves the best validation checkpoint and uses early stopping.

The QB decision evaluation has two parts:

- Receiver choice: compare the actual target to the best receiver at the actual throw frame.
- Timing: compare the actual throw to the best receiver-frame before or at the throw.

The analysis reports exact best-choice rate and acceptable-choice rate. A choice is acceptable if it is within a dynamic range of the best option: `max(1.0 expected yard, 20% of the best option)`.

The calibrated value formula uses the adjusted completion and interception probabilities:

```text
expected yards = calibrated completion probability * predicted completed yards
              + calibrated interception probability * interception yard penalty
```

## Results

The tree model used 6,365 targeted passing plays and 1,102,988 receiver-frame rows. The train/test split was 5,091 targeted throw rows for training and 1,274 for testing.

Tree model test metrics:

- Completion AUC: 0.786
- Completion Brier score: 0.176
- Interception AUC: 0.788
- Interception Brier score: 0.023
- Total yards MAE: 3.919
- Air yards MAE: 0.623
- YAC MAE: 3.865

Graph model best validation metrics after the latest task-specific run:

- Outcome model best epoch: 1
- Outcome model best Brier average: 0.110
- Air-yards model best epoch: 10
- Air-yards model MAE: 3.478
- YAC model best epoch: 19
- YAC model MAE: 2.828

The graph model improved, especially for interception risk and YAC, but XGBoost remained better on the final component comparisons for completion, interception, air yards, and YAC. The graph model is useful as an experimental movement model, but XGBoost remains the main decision scorer.

QB decision results across all evaluated throws:

- Best receiver choice rate: 20.1%
- Acceptable receiver choice rate: 35.6%
- Average missed expected yards: 3.71
- Right receiver and right time rate: 4.6%
- Acceptable timing rate: 13.7%
- Average timing loss: 6.45 expected yards

Among QBs with at least 100 throws, the highest exact best-choice rates belonged to Tua Tagovailoa, Ryan Tannehill, Lamar Jackson, Baker Mayfield, and Patrick Mahomes. The strongest acceptable-choice rates belonged to Lamar Jackson, Cooper Rush, Justin Fields, Tua Tagovailoa, and Marcus Mariota.

## Limitations

The biggest limitation is target selection bias. The model is trained on actual targets, but the decision task asks about non-targeted receivers. That means non-targeted receiver value is a counterfactual estimate.

The model does not fully know the quarterback's read progression. A receiver may look open in tracking data but may not have been in the quarterback's realistic field of vision.

Air yards error is very low for the tree model, so target-point features should be reviewed for possible leakage. The model should use the same opportunity definition for actual targets and non-targeted receivers.

The graph model improved, but it still needs more tuning. It likely needs better calibration, cleaner validation splits, and more careful regularization.

The results should not be treated as a final scouting grade by themselves. They are best used to flag plays for film review.

## Future Work

Future extensions:

- Add read-progression and QB vision constraints.
- Use cleaner counterfactual labels for non-targeted receivers.
- Validate by held-out weeks as the main split.
- Review target-point features for leakage.
- Tune the graph model with better calibration, regularization, and validation splits.
- Build play-level clips or charts for the highest missed-value examples.
- Add model calibration checks by pressure, coverage, and throw depth.
- Combine the model outputs with film review from scouts.
