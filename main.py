# main files for the project

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")


user_input = input("Enter your text: ")

# Tokenize the input
inputs = tokenizer(user_input, return_tensors="pt")

# Generate a response
output = model.generate(**inputs, max_length=50, num_return_sequences=1)

# Decode and print the response
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Response:", response)