import os, logging, time
import torch
import numpy as np
from typing import Tuple

from iris_proj import linear_mlp

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

Device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#logging.info(f"Device: {Device}")


#lets load the model weights 
def load_iris_weights(weight_path: str, in_features: int)-> torch.nn.Module:
    model = linear_mlp(in_features) #our linear_mlp model on takes 1 input(input dimensions)

    state = torch.load(weight_path, map_location=Device, weights_only=True)

    model.load_state_dict(state) #load the weights now
    model.eval() #enter eval

    return model.to(Device) #put the model on the gpu


#Now lets setup the inference to make the predictions
@torch.inference_mode()
def forward_classify(model: torch.nn.Module, x:np.ndarray) -> tuple[int, list[float]]:
    # x: shape(1,4) numpy -> returns a list

    t = torch.from_numpy(x).to(torch.float32).to(Device) 
    logits = model(t)   #[1,3]

    prob = torch.softmax(logits, dim=1)     #[1,3]
    pred = int(prob.argmax(dim=1).item())   #scalar

    return pred, prob[0].tolist() #list[3]

#now for warmup to catch errors and initial testing
def warmup(model: torch.nn.Module, n_features: int)-> None:
    
    fake = torch.zeros((1, n_features), dtype=torch.float32, device=Device)
    with torch.inference_mode():
        _ = torch.softmax(model(fake).view(-1), dim=0)
