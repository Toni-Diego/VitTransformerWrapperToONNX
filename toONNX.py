import torch
from torch.autograd import Variable
import torch.onnx
from vit import ViTransformerWrapper
from vit import get_encoder

# Define los parámetros para el modelo. Asegúrate de que estos coincidan con los utilizados durante el entrenamiento.
args = {
    'max_width': 672,
    'max_height': 192,
    'channels':1,
    'patch_size': 16,
    'dim': 256,
    'encoder_depth': 4,
    'heads': 8
}

# Crea una instancia del modelo
model = get_encoder(args)

dummy_input = Variable(torch.randn(1, 1, 192, 672))

# Mapea los pesos a la CPU
state_dict = torch.load('weights.pth', map_location=torch.device('cpu'))

# Carga los pesos en el modelo
model.load_state_dict(state_dict, strict=False)

# Exporta el modelo a ONNX
torch.onnx.export(model, dummy_input, "moment-in-time.onnx")