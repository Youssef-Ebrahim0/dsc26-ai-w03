# AI Usage

## Tools I used, and for what

I used ChatGPT to help implement and understand the customer history features, chronological time split, and preprocessing pipeline. I also used it to review edge cases, suggest focused tests, and understand evaluation metrics. I reviewed the generated suggestions, made the final code changes, and ran the tests and performance checks myself.

## AI-suggested code I rejected or fixed

An earlier version grouped customer parcels by their exact `created_at` timestamp. After reviewing the grading requirement that parcels created within the same minute must not count as prior parcels for one another, I changed the grouping logic to use `created_at.floor("min")` instead.

I added `test_history_same_minute_parcels_do_not_count_each_other` to verify that parcels with different seconds in the same minute do not see each other as previous parcels. The test passed.

## How I checked that my model has no leakage

I used only features available when a parcel is created and excluded post-event information such as `courier_notes`, `delivered_at`, and the target column `failed`. Customer history uses only earlier minute groups, and model evaluation uses a chronological time split. I also ran the local tests and checked performance against the dummy baseline on multiple seeds.