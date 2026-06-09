# Outputs

The project creates model outputs with `expected_receiver_yards.py` and analysis outputs with `analyze_expected_yards.py`.

## Expected Receiver Yards Outputs

By default, the main pipeline writes to `expected_yards_output_all_weeks/`. The output directory can be changed with `--output-dir` or `make OUTPUT_DIR=...`.

### `receiver_opportunity_features.csv`

Frame-level, model-ready data for every eligible route runner before the throw or play-ending event.

Important fields include:

- Play identifiers: `gameId`, `playId`, `frameId`, `nflId`, `displayName`
- Labels: `wasTargettedReceiver`, `hadPassReception`, `is_interception`, `receivingYards`, `actual_epa`
- Time and context: `seconds_since_snap`, `down`, `yardsToGo`, `pre_snap_ep`
- Receiver/QB geometry: `receiver_x`, `receiver_y`, `qb_x`, `qb_y`, `throw_distance`
- Pressure and lane features: `nearest_rusher_distance`, `rushers_within_3_yards`, `throwing_lane_defender_count`
- Extra pressure fields: `time_to_pressure`, `pressure_score`, `is_under_pressure`
- Catch-point features: `catch_point_nearest_defender_distance`, `catch_point_defender_arrival_margin`, `catch_point_blocker_advantage_5`

### `receiver_expected_yards.csv`

The scored receiver-frame table. It contains all feature rows plus model predictions:

- `completion_probability`
- `completion_probability_bayes`
- `completion_probability_p25`
- `completion_probability_p50`
- `completion_probability_p75`
- `completion_probability_low`
- `completion_probability_high`
- `interception_probability`
- `predicted_yards_if_completed`
- `predicted_yards_if_completed_p25`
- `predicted_yards_if_completed_p50`
- `predicted_yards_if_completed_p75`
- `predicted_air_yards_if_completed`
- `predicted_yac_if_completed`
- `completion_expected_yards`
- `interception_yards_risk`
- `risk_adjusted_expected_yards`
- `expected_yards_p25`
- `expected_yards_p50`
- `expected_yards_p75`
- `expected_yards`
- `predicted_epa`

Use this file when studying receiver opportunity value frame by frame.

### `play_decision_summary.csv`

One row per play comparing the actual target against the highest-value available options at the throw frame.

Important fields include:

- `actual_target_name`
- `actual_target_expected_yards`
- `actual_target_expected_yards_p25`
- `actual_target_expected_yards_p50`
- `actual_target_expected_yards_p75`
- `actual_target_predicted_epa`
- `best_yards_option_p25_name`
- `best_yards_option_p50_name`
- `best_yards_option_p75_name`
- `best_available_expected_yards_p25`
- `best_available_expected_yards_p50`
- `best_available_expected_yards_p75`
- `best_epa_option_name`
- `best_available_predicted_epa`
- `missed_expected_yards_p25`
- `missed_expected_yards_p50`
- `missed_expected_yards_p75`
- `best_timing_option_p25_name`
- `best_timing_option_p50_name`
- `best_timing_option_p75_name`
- `timing_loss_expected_yards_p25`
- `timing_loss_expected_yards_p50`
- `timing_loss_expected_yards_p75`
- `right_receiver_right_time_p25`
- `right_receiver_right_time_p50`
- `right_receiver_right_time_p75`
- `acceptable_yards_choice_p25`
- `acceptable_yards_choice_p50`
- `acceptable_yards_choice_p75`
- `acceptable_timing_choice_p25`
- `acceptable_timing_choice_p50`
- `acceptable_timing_choice_p75`
- `result_yards_over_expected`
- `result_epa_over_predicted`
- `missed_predicted_epa`

Positive `missed_expected_yards_p50` means another available receiver had a higher median model-estimated yardage value than the actual target at the throw frame. Positive `timing_loss_expected_yards_p50` means a higher-value receiver-time opportunity existed before or at the actual throw. The P25 and P75 versions repeat the same questions under conservative and upside assumptions.

### `model_metrics.json`

Model quality and run metadata, including train/test row counts and metrics such as completion AUC, interception AUC, yards MAE, and EPA MAE when available.

## Analysis Outputs

`analyze_expected_yards.py` writes CSVs under `<OUTPUT_DIR>/analysis/`.

- `enriched_play_decision_summary.csv`: play summary joined with team, QB, coverage, down, distance, EPA, and pass-result context.
- `overall_summary.csv`: aggregate decision quality by train/test split.
- `team_summary.csv`: team-level decision quality.
- `qb_summary_all.csv`: quarterback-level decision quality for every QB that appears in the selected data unless `--min-throws` is raised.
- `qb_summary.csv`: the same quarterback metrics split by train/test.
- `coverage_summary.csv`: decision quality by man/zone coverage label.
- `pressure_lane_summary.csv`: decision quality by nearest-rusher and throwing-lane buckets.
- `time_curve_summary.csv`: receiver value evolution by time since snap.
- `top_missed_plays_test.csv`: highest missed-value test plays.
- `top_missed_plays_train.csv`: highest missed-value train plays.
- `model_error_comparison.csv`: model MAE by target for the tree and graph models.
- `model_error_comparison.png`: grouped bar chart of the same model errors.
- `completion_calibration.csv`: predicted completion buckets vs actual completion rate.
- `interception_calibration.csv`: predicted interception buckets vs actual interception rate.
- `figures/`: extra QB and model graphics.

## Graph Transformer Outputs

The optional graph transformer writes to `graph_transformer_smoke/` by default when run through Makefile smoke targets:

- `spatiotemporal_graph_transformer.pt`: model weights.
- `feature_scaler.pkl`: fitted feature scaler.
- `metadata.json`: model configuration and training history.
- `graph_receiver_opportunities.csv`: inference output when running graph inference.
- `graph_play_decision_summary.csv`: graph-model target choice summary.

The graph metadata includes the best validation epoch, prediction task, and the validation metric used for checkpoint selection.
