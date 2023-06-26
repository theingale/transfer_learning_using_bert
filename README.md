# transfer_learning_using_bert
Applied Transfer Learning technique on amazon review text data using pretrained BERT Model to predict if the given review is positive or negative.


## Approach Followed:
1. Getting the reviews text data and their labels.
2. Cleaning (preprocessing) the review text by removing html tokens present (if any) in the review text.
3. Tokenizing the review text by using Tokenizer class in the tokenization.py file.
4. Adding BERT special tokens like [CLS], [SEP], [PAD] to the text tokens and converting them into token IDs. Also creating mask_input and segment_input arrays for each sequence of tokens.
5. Creating pretrained bert model using tf.hub and keras layer.
6. Giving 3 inputs, token_ids, mask_input and segment_input to bert model and getting predicted 768 dimensional BERT encodings corresponding to the [CLS] token for each input.
7. A simple feed forward neural network with dense layers and sigmoid activation output with input as BERT 768 dimensional encodings and corresponding labels is trained.
8. This dense model is used to predict the review sentiment as positive [1] or negative [0].
