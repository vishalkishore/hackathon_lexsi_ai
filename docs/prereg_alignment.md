# Alignment cosine (theory test), pre-registered 2026-11-14 before computing. No generation.
Data: fresh_v1 TEST split (198 pairs, matched harmful/benign by group_id). Base weights, bf16 forward, fp32 activations,
hidden_states[19] (output of layers[18]) at the last prompt token (= r_hat layer/token; steering.json layer 19, same token).
Conditions: en, hi-Deva, hi-Latn raw, hi-Latn after N (IndicXlit final message), en-hi raw; depths 0 and 8.
Per condition: d = mean_pairs(h_harm - h_benign); cos_r = cos(d, r_hat); cos_wnew = cos(d, w_new) (Step 5 fit, base);
cos_wold = cos(d, w_old) (calib_w, base); cos_en = cos(d, d_en same depth); gap_r = r_hat . d; ceiling = mean over 200
random half-splits of pairs (seed 0) of cos(d_half1, d_half2). 95% CI: percentile bootstrap over pairs, 4000, seed 0.
P-A cos_r: en > hi-Deva > hi-Latn raw (point estimates) at d0 and at d8.
P-B cos_r(hi-Latn N) lies inside the 95% CI of cos_r(hi-Deva), at d0 and at d8.
P-C hi-Latn N: cos_wnew > cos_wold with non-overlapping 95% CIs, at d0 and at d8.
P-D hi-Latn raw: cos_r(d8) < cos_r(d0) (point estimates).
