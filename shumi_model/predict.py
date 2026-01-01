from inference import predict_next_actions

num_of_predictions = 10

next_actinos = predict_next_actions(num_of_predictions)
for action in next_actinos:
    print(f"Predict next {action[0]} with probabilities {action[1]}")
