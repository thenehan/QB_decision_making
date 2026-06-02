# Technical Summary

## Methods

The project builds receiver-frame opportunities from NFL Big Data Bowl tracking data. For each dropback, eligible route runners are scored on every frame from ball snap through the throw or play-ending event. The feature set includes receiver movement, quarterback location, throw distance, projected catch point, nearest defenders, throwing-lane defenders, catch-point spacing, and pressure fields such as nearest rusher distance, time to pressure, pressure score, and whether the quarterback is under pressure.

The main model is an XGBoost tree model. It predicts completion probability, interception probability, air yards, YAC, and expected yards. Uncertainty is represented with P25, P50, and P75 expected-yards estimates.

The graph model is a GAT-style temporal transformer. It uses player tracking sequences, target-player identity, and play context features. The improved version predicts completion, interception, air yards, YAC, and total yards directly. It saves the best validation checkpoint and uses early stopping.

The QB decision evaluation has two parts:

- Receiver choice: compare the actual target to the best receiver at the actual throw frame.
- Timing: compare the actual throw to the best receiver-frame before or at the throw.

The analysis reports exact best-choice rate and acceptable-choice rate. A choice is acceptable if it is within a dynamic range of the best option: `max(1.0 expected yard, 20% of the best option)`.

## Results

The tree model used 6,366 targeted passing plays and 1,103,047 receiver-frame rows. The train/test split was 5,143 targeted throw rows for training and 1,223 for testing.

Tree model test metrics:

- Completion AUC: 0.786
- Completion Brier score: 0.179
- Interception AUC: 0.728
- Interception Brier score: 0.023
- Total yards MAE: 3.848
- Air yards MAE: 0.634
- YAC MAE: 3.745

Graph model best validation metrics:

- Best epoch: 13
- Total yards MAE: 5.793
- Air yards MAE: 3.461
- YAC MAE: 2.907
- Completion AUC: 0.759
- Interception AUC: 0.828

The tree model still had the better total-yards error, but the graph model had better YAC error. That suggests the graph model may be picking up useful movement information, but it is not yet strong enough to replace the tree model as the main decision scorer.

QB decision results on the test split:

- Best receiver choice rate: 25.2%
- Acceptable receiver choice rate: 40.3%
- Average missed expected yards: 3.46
- Right receiver and right time rate: 5.0%
- Acceptable timing rate: 17.9%
- Average timing loss: 5.89 expected yards

Among QBs with at least 50 throws, the lowest missed expected yards included Lamar Jackson, Patrick Mahomes, Marcus Mariota, P.J. Walker, and Cooper Rush. The best acceptable-choice rates included Lamar Jackson, Jalen Hurts, Aaron Rodgers, Tua Tagovailoa, Zach Wilson, Russell Wilson, Ryan Tannehill, Justin Fields, Marcus Mariota, Patrick Mahomes, Trevor Lawrence, and Josh Allen.

## Limitations

The biggest limitation is target selection bias. The model is trained on actual targets, but the decision task asks about non-targeted receivers. That means non-targeted receiver value is a counterfactual estimate.

The model does not fully know the quarterback's read progression. A receiver may look open in tracking data but may not have been in the quarterback's realistic field of vision.

Air yards error is very low for the tree model, so target-point features should be reviewed for possible leakage. The model should use the same opportunity definition for actual targets and non-targeted receivers.

The graph model improved, but it still needs more tuning. It likely needs more careful architecture choices, validation splits, and possibly route/context features.

The results should not be treated as a final scouting grade by themselves. They are best used to flag plays for film review.

## Future Work

Future extensions:

- Add read-progression and QB vision constraints.
- Use cleaner counterfactual labels for non-targeted receivers.
- Validate by held-out weeks as the main split.
- Review target-point features for leakage.
- Tune the graph model with larger hidden size, longer training, and route features.
- Build play-level clips or charts for the highest missed-value examples.
- Add model calibration checks by pressure, coverage, and throw depth.
- Combine the model outputs with film review from scouts.
