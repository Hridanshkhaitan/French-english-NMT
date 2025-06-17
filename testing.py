# import torch
# from a2_transformer_model import LayerNorm  # Replace `your_module` with the name of your file where LayerNorm is defined

# # Instantiate the LayerNorm module
# num_features = 5  # Dimensionality of the features
# layer_norm = LayerNorm(num_features)

# # Test input tensor of shape [batch_size, seq_len, num_features]
# x = torch.randn(3, 4, num_features)  # Example input with batch_size=3, seq_len=4

# # Forward pass
# output = layer_norm(x)

# # Check the shape of the output
# assert output.shape == x.shape, f"Expected output shape {x.shape}, but got {output.shape}"
# print("Shape test passed.")

# # Check mean and std deviation along the feature dimension
# mean = output.mean(dim=-1)  # Mean over the feature dimension
# std = output.std(dim=-1)    # Std deviation over the feature dimension

# # Verify that the mean is close to 0
# assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), "Mean test failed"
# print("Mean test passed.")

# # Verify that the std deviation is close to 1
# assert torch.allclose(std, torch.ones_like(std), atol=1e-5), "Std deviation test failed"
# print("Standard deviation test passed.")


# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("raeidsaqur/mt_fr2en_hansard_t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("raeidsaqur/mt_fr2en_hansard_t5-small")

tokenizer.src_lang = "fr_XX"  # Source language: French
target_language_code = "en_XX"  # Target language: English

# Define the French sentences to test
french_sentences = [
   "Les conservateurs promettent que s’ils sont elus, vos parents se reuniront, votre emission de television preferee ne sera pas annulee et McDonald’s ramenera la pizza",
    ]

# Translate each French sentence and compare results
for i, sentence in enumerate(french_sentences, 1):
    # Tokenize and translate
    inputs = tokenizer(sentence, return_tensors="pt")
    translated_tokens = model.generate(**inputs)

    # Decode the generated tokens
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print(f"Original Sentence {i}: {sentence}")
    print(f"Translated by T5 Model {i}: {translated_text}\n")