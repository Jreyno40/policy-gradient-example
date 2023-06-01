import json

def convert_model_to_json(model, weight_history):

    json_obj = {
        "nodes": [],
        "links": []
    }

    weight_mapping = {}

    # Create nodes for input neurons
    input_size = model.fc1.in_features
    for i in range(input_size):
        input_node = {
            "id": f"input_{i}",
            "layer": "input",
            # Add any other relevant information to the input node
        }
        json_obj["nodes"].append(input_node)

    # Iterate over model layers and add neurons and edges
    for name, layer in model.named_children():
        # Create nodes for neurons in the current layer
        for neuron_idx in range(layer.out_features):
            neuron_name = f"{name}_{neuron_idx}"
            neuron = {
                "id": neuron_name,
                "layer": name,
                # Add any other relevant information to the neuron
            }
            json_obj["nodes"].append(neuron)

            # Create edges between neurons in the current and previous layer
            if name != "fc1":
                prev_layer_name = f"fc{int(name[-1]) - 1}"
                for prev_neuron_idx in range(model.__getattr__(prev_layer_name).out_features):
                    prev_neuron_name = f"{prev_layer_name}_{prev_neuron_idx}"
                    edge = {
                        "source": prev_neuron_name,
                        "target": neuron_name,
                        # Add any other relevant information to the edge
                    }

                    link_name = f"{prev_neuron_name}_{neuron_name}"
                    weight_mapping[link_name] = []

                    json_obj["links"].append(edge)

        # Create edges between input neurons and neurons in the first layer
        if name == "fc1":
            for neuron_idx in range(layer.out_features):
                neuron_name = f"{name}_{neuron_idx}"
                for input_idx in range(input_size):
                    input_name = f"input_{input_idx}"
                    edge = {
                        "source": input_name,
                        "target": neuron_name,
                        # Add any other relevant information to the edge
                    }

                    link_name = f"{input_name}_{neuron_name}"
                    weight_mapping[link_name] = []

                    json_obj["links"].append(edge)

    for i, weight_episode in enumerate(weight_history):            
            for link in json_obj["links"]:
                source_neuron = link["source"]
                target_neuron = link["target"]
                source_layer, source_idx = source_neuron.split("_")
                _, target_idx = target_neuron.split("_")
                
                if(source_layer == 'input'):
                    weight_tensor = weight_episode['fc1.weight']
                elif(source_layer == 'fc1'):
                    weight_tensor = weight_episode['fc2.weight']
		
                weight = weight_tensor[int(target_idx)][int(source_idx)]
                link["weight_episode_" + str(i)] = weight.item()

    with open("model.json", "w") as outfile:
        json.dump(json_obj, outfile)


def convert_rewards_to_json(reward_track):
    reward_dict = {i : reward_track[i] for i in range(len(reward_track)) }

    with open("rewards.json", "w") as outfile:
        json.dump(reward_dict, outfile)