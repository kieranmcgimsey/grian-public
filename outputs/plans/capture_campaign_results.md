# Capture campaign — headline results

Append-only. One row per (model, executor, window). See capture_campaign.md §6.4.

| date | model | executor | window | capture | revenue | oracle | spearman | top10 share | sha |
|---|---|---|---|---|---|---|---|---|---|
| 2026-07-11 | naive_similar_day | open-loop | val | 0.452 | $5,415,502 | $11,994,630 | 0.566 | 0.517 | 4ed7511 |
| 2026-07-11 | autoregression | open-loop | val | 0.473 | $5,677,574 | $11,994,630 | 0.665 | 0.517 | 4ed7511 |
| 2026-07-11 | lightgbm_rich | open-loop | val | 0.389 | $4,665,826 | $11,994,630 | 0.632 | 0.517 | 4ed7511 |
| 2026-07-11 | naive_similar_day | open-loop | test | 0.347 | $2,676,924 | $7,707,632 | 0.575 | 0.307 | 4ed7511 |
| 2026-07-11 | autoregression | open-loop | test | 0.402 | $3,095,584 | $7,707,632 | 0.668 | 0.307 | 4ed7511 |
| 2026-07-11 | lightgbm_rich | open-loop | test | 0.301 | $2,320,346 | $7,707,632 | 0.688 | 0.307 | 4ed7511 |
| 2026-07-11 | naive_similar_day | mpc-30m | val | 0.382 | $4,583,597 | $11,994,630 | 0.566 | 0.517 | bb8b431 |
| 2026-07-11 | lightgbm_rich | mpc-30m | val | 0.536 | $6,429,456 | $11,994,630 | 0.839 | 0.517 | bb8b431 |
| 2026-07-11 | naive_similar_day | mpc-30m | test | 0.365 | $2,810,971 | $7,707,632 | 0.575 | 0.307 | 23b0c71 |
| 2026-07-11 | lightgbm_rich | mpc-30m | test | 0.508 | $3,919,430 | $7,707,632 | 0.826 | 0.307 | 23b0c71 |
| 2026-07-11 | naive_similar_day | open-loop | val | 0.452 | $5,415,502 | $11,994,630 | 0.566 | 0.517 | e5b1ebc |
| 2026-07-11 | autoregression | open-loop | val | 0.473 | $5,677,574 | $11,994,630 | 0.665 | 0.517 | e5b1ebc |
| 2026-07-11 | lightgbm_rich | open-loop | val | 0.389 | $4,665,826 | $11,994,630 | 0.632 | 0.517 | e5b1ebc |
| 2026-07-11 | lightgbm_qmean | open-loop | val | 0.407 | $4,883,029 | $11,994,630 | 0.672 | 0.517 | e5b1ebc |
| 2026-07-11 | naive_similar_day | open-loop | test | 0.347 | $2,676,924 | $7,707,632 | 0.575 | 0.307 | e5b1ebc |
| 2026-07-11 | autoregression | open-loop | test | 0.402 | $3,095,584 | $7,707,632 | 0.668 | 0.307 | e5b1ebc |
| 2026-07-11 | lightgbm_rich | open-loop | test | 0.301 | $2,320,346 | $7,707,632 | 0.688 | 0.307 | e5b1ebc |
| 2026-07-11 | lightgbm_qmean | open-loop | test | 0.443 | $3,411,361 | $7,707,632 | 0.7 | 0.307 | e5b1ebc |
| 2026-07-11 | lightgbm_qmean | mpc-30m | val | 0.520 | $6,232,304 | $11,994,630 | 0.829 | 0.517 | d7abbec |
| 2026-07-12 | lightgbm_rich | mpc-r6_f6_p0 | val | 0.546 | $6,554,161 | $11,994,630 | 0.869 | 0.517 | 823ef06 |
| 2026-07-12 | lightgbm_rich | mpc-r1_f6_p0 | val | 0.545 | $6,542,162 | $11,994,630 | 0.869 | 0.517 | 823ef06 |
| 2026-07-12 | lightgbm_rich | mpc-r6_f12_p6 | val | 0.435 | $5,217,956 | $11,994,630 | 0.839 | 0.517 | 823ef06 |
| 2026-07-12 | lightgbm_rich | mpc-r1_f6_p6 | val | 0.428 | $5,130,637 | $11,994,630 | 0.869 | 0.517 | 823ef06 |
| 2026-07-12 | lightgbm_rich | mpc-r6_f6_p0 | val | 0.546 | $6,554,161 | $11,994,630 | 0.869 | 0.517 | 9c7e386 |
| 2026-07-12 | lightgbm_rich | mpc-r1_f6_p0 | val | 0.545 | $6,542,162 | $11,994,630 | 0.869 | 0.517 | 9c7e386 |
| 2026-07-12 | lightgbm_rich | mpc-r6_f12_p6 | val | 0.435 | $5,217,956 | $11,994,630 | 0.839 | 0.517 | 9c7e386 |
| 2026-07-12 | lightgbm_rich | mpc-r1_f6_p6 | val | 0.428 | $5,130,637 | $11,994,630 | 0.869 | 0.517 | 9c7e386 |
| 2026-07-12 | lightgbm_rich | mpc-r1_f6_p6_g300 | val | 0.523 | $6,272,930 | $11,994,630 | 0.869 | 0.517 | 9c7e386 |
| 2026-07-12 | lightgbm_rich | mpc-r6_f6_p0 | test | 0.562 | $4,332,464 | $7,707,632 | 0.864 | 0.307 | 1ab363b |
| 2026-07-12 | lightgbm_rich | mpc-r6_f6_p0 | val | 0.546 | $6,554,161 | $11,994,630 | 0.869 | 0.517 | 0f9ce41 |
| 2026-07-12 | lightgbm_rich | mpc-r1_f6_p0 | val | 0.545 | $6,542,162 | $11,994,630 | 0.869 | 0.517 | 0f9ce41 |
| 2026-07-12 | lightgbm_rich | mpc-r6_f12_p6 | val | 0.435 | $5,217,956 | $11,994,630 | 0.839 | 0.517 | 0f9ce41 |
| 2026-07-12 | lightgbm_rich | mpc-r1_f6_p6 | val | 0.428 | $5,130,637 | $11,994,630 | 0.869 | 0.517 | 0f9ce41 |
| 2026-07-12 | lightgbm_rich | mpc-r1_f6_p6_g300 | val | 0.523 | $6,272,930 | $11,994,630 | 0.869 | 0.517 | 0f9ce41 |
| 2026-07-12 | lightgbm_rich | mpc-r1_f6_p6_g1000 | val | 0.544 | $6,524,237 | $11,994,630 | 0.869 | 0.517 | 0f9ce41 |
| 2026-07-12 | lightgbm_rich | mpc-r1_f1_p0 | val | 0.534 | $6,410,614 | $11,994,630 | 0.895 | 0.517 | 0f9ce41 |
| 2026-07-12 | lightgbm_rich[rich150] | mpc-r6_f6 | val | 0.548 | $6,576,866 | $11,994,630 | 0.866 | 0.517 | b6217ed |
| 2026-07-12 | lightgbm_qmean[qmean150] | mpc-r6_f6 | val | 0.539 | $6,471,570 | $11,994,630 | 0.86 | 0.517 | b6217ed |
| 2026-07-12 | lightgbm_qmean[qhybrid150] | mpc-r6_f6 | val | 0.522 | $6,265,500 | $11,994,630 | 0.864 | 0.517 | b6217ed |
| 2026-07-12 | lightgbm_rich[baseline] | mpc-r6_f6 | val | 0.546 | $6,554,161 | $-- | 0.869 | 0.517 | a299221 |
| 2026-07-12 | lightgbm_rich[scarcity] | mpc-r6_f6 | val | 0.548 | $6,569,357 | $-- | 0.869 | 0.517 | a299221 |
