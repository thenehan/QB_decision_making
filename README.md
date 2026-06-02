# QB Decision Making Project

## Project Idea

This project is about quarterback decision making. The goal is not just to ask whether the pass was completed. The goal is to ask a more scouting-focused question:

Did the quarterback throw to the right receiver at the right time?

On every passing play, the code looks at each eligible receiver before the throw. It estimates how good of an option each receiver was at that moment. Then it compares the quarterback's actual target to the other receivers who were available.

In football terms, this is trying to separate:

- a good decision with a bad throw
- a bad decision that still worked
- a safe decision that passed up a better option
- a late throw where the window was better earlier
- an aggressive throw that was actually worth the risk

## Executive Summary

The main XGBoost model evaluated 6,366 targeted passing plays and more than 1.1 million receiver-frame opportunities from Weeks 1-9. On the test set, the model predicted completion with a 0.786 AUC -this is a good predictive score shows the model is learning while a 0.5 AUC would be more of a coinflip model- and total receiving yards with a 3.85 MAE meaning on average our predicted yards are off by less then 4 yards. I also tried using a GAT model with a temporal transformer, but XGBoost still had the better total-yards error. The graph model was better at predicting YAC, which suggests it is learning some movement information, but it is not ready to replace the XGBoost model.

On the test split, quarterbacks threw to the model's best receiver about 25.2% of the time. When close choices are counted with a dynamic acceptable range, that rises to 40.3%. The average missed value was about 3.5 expected yards at the actual throw frame. Timing was stricter: the right receiver at the right time happened about 5.0% of the time, while acceptable timing happened about 17.9% of the time.

This does not mean quarterbacks are usually wrong as there are factors that aren't entirely quantitative such as yards to down, play design ie what was the progression like on that play and interception percentage (due to the small number of interceptions this part of the model and the detraction of yards is a bit more variable). But this model can tell us that it believes there is another receiver or another moment to throw that looked better and this can be of real value as a film-study flag, not as an automatic negative grade.

Among QBs with at least 50 throws, Lamar Jackson, Patrick Mahomes, Marcus Mariota, P.J. Walker, and Cooper Rush had some of the lowest missed expected yards.  Their acceptable-choice rates were 52.0% for Jackson, 43.4% for Mahomes, 43.7% for Mariota, 42.3% for Walker, and 40.7% for Rush. The strongest acceptable-choice rates (acceptable choce being a linear regression band being variable to the "correct option") overall included Lamar Jackson at 52.0%, Jalen Hurts at 48.1%, Aaron Rodgers at 47.6%, Tua Tagovailoa at 47.2%, Zach Wilson at 44.2%, Russell Wilson at 44.0%, Ryan Tannehill at 43.9%, Justin Fields at 43.8%, Marcus Mariota at 43.7%, Patrick Mahomes at 43.4%, Trevor Lawrence at 43.2%, and Josh Allen at 43.1%. These results are interesting as it shows the model is valueing more risky play then tradional metrics this can be because of the inefficencies of the data pipeline, such as making the interception pipelien more robust but it can also show that some quarterbacks are making correct decsions but are not throwing the ball accurate enough to result in great plays. 

The best use of this project is to identify plays where the model and the quarterback disagreed, then watch the film. The scout can ask: was the better option actually visible, did pressure force the throw, was the quarterback working a designed read, or did he really miss a better window?

## Metric Explanation

### Expected yards

Expected yards is the model's estimate of how many yards a throw option is worth before the ball is thrown.

Example:

```text
Receiver A: 6 expected yards
Receiver B: 11 expected yards
Actual throw: Receiver A
Missed expected yards: 5
```

That does not automatically mean the QB made a bad play. It means Receiver B looked like a better option based on the tracking data, so that play is worth reviewing.

### Best choice rate

Best choice rate asks:

How often did the quarterback throw to the model's highest-valued receiver?

This is a strict metric. If the actual target was worth 9.8 expected yards and another receiver was worth 10.0, the quarterback did not get credit for the best choice.

### Acceptable choice rate

Acceptable choice rate is more forgiving. It gives the quarterback credit if the target was close enough to the best option.

In this project, a throw is acceptable if it is close to the best receiver using a dynamic range. The range is at least 1 expected yard, or 20% of the best option, whichever is larger.

Example:

```text
Best receiver: 10.0 expected yards
Acceptable range: 8.0 expected yards or more
Actual target: 9.4 expected yards
Result: acceptable decision
```

For a short throw, this stays tight. If the best option is 5 expected yards, the actual target needs to be within 1 yard. For a bigger shot, the range gets wider. If the best option is 25 expected yards, an option around 20 expected yards still counts as acceptable. This is probably more useful for scouting than best choice rate because football decisions are often close, and not every deep option has to be matched yard-for-yard.

### Timing loss

Timing loss asks whether the quarterback missed a better throwing window.

Example:

```text
Receiver value at 1.8 seconds: 12 expected yards
Receiver value at actual throw: 7 expected yards
Timing loss: 5 expected yards
```

This might mean the QB was late, or it might mean the read progression did not get to that receiver in time. The model flags the play, and film review explains why.

### Result over expected

Result over expected separates the decision from the throw/result.

Example:

```text
Expected value before throw: 6 yards
Actual result: 18 yards
Result over expected: +12 yards
```

That could mean the receiver made a great play after the catch, the throw was excellent, or the defense made a mistake. It is different from saying the original decision was the best available decision.

## Simple Play Examples

### Example 1: Completed pass, but possibly not the best decision

A quarterback completes a checkdown for 5 yards. The box score says the play worked. But the model may see a receiver breaking open downfield worth 12 expected yards. This would show up as a positive missed expected yards number.

The scouting question is:

Did the quarterback correctly take the checkdown because of pressure or read progression, or did he miss the bigger window?

### Example 2: Incomplete pass, but good decision

A quarterback throws to a receiver with separation near the sticks, but the pass is dropped. The result is incomplete, but the model may still rate the target as the best option. This protects the quarterback from being blamed for a bad result that came from a reasonable decision.

### Example 3: Right receiver, wrong time

A receiver is open early in the route, but the quarterback throws two beats later after the defender recovers. The actual target can be right, but the timing can still be off.

That is why this project evaluates both receiver choice and throw timing.

## Main Questions

1. Did the QB throw to the right receiver?
2. Did the QB throw at the right time?
3. How much does the answer change when prediction uncertainty is included?
4. Does a graph attention model beat a tree model on air yards and YAC prediction?
5. Which decisions were close enough to count as acceptable?

## Models Used

The first model is an XGBoost tree model. This is the main model used for the decision analysis.

It predicts:

- completion probability
- interception probability
- air yards
- yards after catch
- expected yards

It also uses football context like:

- nearest rusher distance
- time to pressure
- throwing lane defenders
- receiver location
- defender spacing
- down and distance

The second model is a graph attention plus temporal transformer model. It looks at player locations over time. This is closer to how a play unfolds visually, because it sees the movement of all players across multiple frames.

The graph model is useful, but in the current results it is still experimental. It did better on YAC error, but XGBoost did better on total yards error. For the final scouting interpretation, XGBoost is the main model.

## Uncertainty

The model keeps three versions of expected yards:

- `p25`: lower estimate
- `p50`: middle estimate
- `p75`: higher estimate

This matters because the model is not perfectly sure. A throw may look questionable under the upside estimate but reasonable under the lower estimate.

Football example:

```text
Safe option: 6 to 8 expected yards
Risky option: 3 to 14 expected yards
```

The safe option may be better under the lower estimate, while the risky option may be better under the higher estimate. This helps describe QB style: safe, aggressive, or balanced.

In the test results, the quarterback decision rates changed depending on which estimate was used:

```text
P25 lower estimate:
Best receiver choice rate: 23.5%
Acceptable receiver choice rate: 36.3%
Average missed expected yards: 3.0

P50 middle estimate:
Best receiver choice rate: 25.2%
Acceptable receiver choice rate: 40.3%
Average missed expected yards: 3.5

P75 higher estimate:
Best receiver choice rate: 25.5%
Acceptable receiver choice rate: 42.3%
Average missed expected yards: 3.6
```

This means the overall conclusion is pretty stable. Even when the model uses the lower estimate, QBs still threw to the exact best receiver only about 23.5% of the time. When close decisions are counted as acceptable with the dynamic range, the number is about 36% to 42% of throws across the three estimates.

Timing changes more across the uncertainty range:

```text
P25 lower estimate:
Right receiver and right time: 8.1%
Acceptable timing: 22.2%

P50 middle estimate:
Right receiver and right time: 5.0%
Acceptable timing: 17.9%

P75 higher estimate:
Right receiver and right time: 5.4%
Acceptable timing: 20.5%
```

The lower estimate is more forgiving on timing because it shrinks some of the high-upside windows. The dynamic acceptable range also makes the timing grade more realistic: missing a 25-yard window by 2 yards is not treated the same as missing a 5-yard window by 2 yards.

## QB Evaluation

For each play, the code checks:

- who the quarterback threw to
- who the model liked best at the throw frame
- whether the actual target was close enough to count as acceptable
- whether there was a better throwing window earlier
- how much expected yardage was missed
- how the actual result compared to the model's expectation

The main QB files are:

- `qb_summary_all.csv`
- `qb_summary.csv`
- `enriched_play_decision_summary.csv`
- `top_missed_plays_test.csv`

`qb_summary_all.csv` is the best starting point for comparing quarterbacks. `top_missed_plays_test.csv` is the best starting point for film review.

## How Scouts Could Use This

This project is not meant to replace film. It is meant to help choose which plays to watch.

Good scouting uses would be:

- find QBs who consistently choose acceptable targets
- find plays where the QB may have missed a better receiver
- separate decision quality from throw quality
- study whether pressure explains missed opportunities
- compare aggressive QBs to safer QBs
- find receivers who were open but not targeted

The model should be treated like an assistant coach saying, "This play is worth another look."

## Files

```text
.
├── expected_receiver_yards.py
├── analyze_expected_yards.py
├── spatiotemporal_graph_transformer.py
├── compare_model_errors.py
├── graph_decision_summary.py
├── make_visuals.py
├── final_report.ipynb
├── docs/
│   ├── DATA_DEPENDENCIES.md
│   ├── OUTPUTS.md
│   └── TECHNICAL_SUMMARY.md
├── Makefile
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The NFL CSV files should be in the data folder you pass with `DATA_DIR`. If this folder is inside the original project folder, this usually works:

```bash
make expected-yards DATA_DIR=..
```

## Run The Project

Run the tree model on all weeks:

```bash
make expected-yards DATA_DIR=..
```

Use held-out weeks instead of a random play split:

```bash
python expected_receiver_yards.py --data-dir .. --output-dir expected_yards_output_all_weeks --weeks 1 2 3 4 5 6 7 8 9 --test-weeks 8 9
```

Make the QB analysis tables:

```bash
make analyze DATA_DIR=..
```

Train the graph model:

```bash
make graph-train DATA_DIR=..
```

Run graph inference and summarize graph choices:

```bash
make graph-infer DATA_DIR=..
make graph-decisions
```

Compare model errors:

```bash
make compare-errors
```

Make the extra visuals:

```bash
make visuals
```

## Quick Test

For a smaller test run:

```bash
make smoke DATA_DIR=..
make analyze-smoke DATA_DIR=..
make graph-smoke DATA_DIR=..
```

## Output

The tree model writes:

- `receiver_opportunity_features.csv`
- `receiver_expected_yards.csv`
- `play_decision_summary.csv`
- `model_metrics.json`

The analysis step writes files under:

```text
expected_yards_output_all_weeks/analysis/
```

The model comparison step writes:

- `model_error_comparison.csv`
- `model_error_comparison.png`
- `completion_calibration.csv`
- `interception_calibration.csv`

The extra visuals are saved in:

```text
expected_yards_output_all_weeks/analysis/figures/
```

The final report notebook is:

```text
final_report.ipynb
```

## Limitations

The biggest limitation is that the model learns from actual targeted throws. When it grades a receiver who was not targeted, it is estimating what might have happened. That is useful, but it is not the same as knowing for sure.

The model also does not fully understand the quarterback's eyes, reads, coaching rules, or whether the quarterback could realistically see the receiver. A receiver can look open in tracking data but still not be a realistic option.

Because of that, the results should be used as a film-study guide, not a final scouting grade.

## Notes

Raw tracking files are not included because they are very large. The `.gitignore` file keeps the raw data, output folders, and model checkpoints out of GitHub.
