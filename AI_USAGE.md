# AI usage

## Tools I used, and for what

I used ChatGPT to help implement `add_history_features` and `time_split`, and to understand and choose suitable parameters for the preprocessing and Logistic Regression pipeline in `build_pipeline`. I implemented the remaining parts myself, including evaluation and baseline comparison, and reviewed and tested the final solution.

## AI-suggested code I rejected or fixed in add_history_features or time_split

I reviewed the history-feature logic to ensure that parcels belonging to the same customer with identical `created_at` timestamps do not count each other as previous parcels. I added `test_same_timestamp_parcels_do_not_count_each_other` to verify this behavior, and the test passed.

## How I checked that my model has no leak

I used only features available when a parcel is created, excluded post-delivery information such as `courier_notes` and `delivered_at`, and evaluated the model on the last 20% of parcels using a chronological time split.