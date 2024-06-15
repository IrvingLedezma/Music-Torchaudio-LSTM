# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:03:23 2024

@author: irvin
"""

import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
os.chdir('C:/Users/irvin/OneDrive/Desktop/python_scripts/deep_learning/proyecto')
    

#===========================================================
# Archivo de configuración
#===========================================================

config = {
    "model": {
        "input_size": 1,  # Tamaño de la entrada del modelo (solo se utiliza 1 característica, el precio de cierre)
        "num_lstm_layers": 2,  # Número de capas LSTM en el modelo
        "lstm_size": 32,  # Tamaño de las celdas LSTM
        "dropout": 0.2,  # Proporción de unidades de salida que se deben dejar fuera durante el entrenamiento
    },
    "training": {
        "device": "cuda",  # Dispositivo de cómputo a utilizar ("cpu" o "cuda" para GPU)
        "batch_size": 64,  # Tamaño del lote utilizado durante el entrenamiento
        "num_epoch": 100,  # Número de épocas de entrenamiento
        "learning_rate": 0.01,  # Tasa de aprendizaje utilizada por el optimizador
        "scheduler_step_size": 40,  # Frecuencia con la que se debe reducir la tasa de aprendizaje durante el entrenamiento
    }
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#===========================================================
# Funciones y Clase para cargar mp3 a tensores torch
#===========================================================

def load_audio(file_path, device):
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = waveform.to(device)
    return waveform, sample_rate

class AudioDataset(Dataset):
    def __init__(self, tensor, look_back, channel):
        self.tensor = tensor
        self.look_back = look_back
        self.channel = channel

    def __len__(self):
        return self.tensor.shape[1] - self.look_back - 1

    def __getitem__(self, idx):
        x = self.tensor[self.channel, idx:idx + self.look_back].unsqueeze(-1)
        y = self.tensor[self.channel, idx + self.look_back]
        return x, y

def create_batches(tensor, look_back, batch_size, channel):
    dataset = AudioDataset(tensor, look_back, channel)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader
        
#===========================================================
# Clase de red neuronal LSTM
#===========================================================

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.5):
        """
        Inicializa el modelo LSTM.

        Args:
            input_size (int, optional): Tamaño de la entrada. Por defecto es 1.
            hidden_layer_size (int, optional): Tamaño de la capa oculta. Por defecto es 32.
            num_layers (int, optional): Número de capas LSTM. Por defecto es 2.
            output_size (int, optional): Tamaño de la salida. Por defecto es 1.
            dropout (float, optional): Tasa de dropout. Por defecto es 0.2.
        """
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        # Capa lineal para transformar la entrada al tamaño de la capa oculta
        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()  # Función de activación ReLU

        # Capa LSTM
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=self.hidden_layer_size, num_layers=num_layers, batch_first=True)

        self.dropout = nn.Dropout(dropout)  # Dropout para regularización

        # Capa lineal para transformar la salida al tamaño deseado
        self.linear_2 = nn.Linear(num_layers*hidden_layer_size, output_size)
        
        self.init_weights()  # Inicialización de los pesos de la capa LSTM

    def init_weights(self):
        """
        Inicializa los pesos de la capa LSTM.
        """
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)  # Inicialización de los sesgos con ceros
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)  # Inicialización de los pesos de entrada utilizando Kaiming normal
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)  # Inicialización de los pesos de la capa oculta utilizando orthogonal

    def forward(self, x):
        """
        Realiza una pasada hacia adelante a través del modelo.

        Args:
            x (Tensor): Datos de entrada.

        Returns:
            Tensor: Predicciones del modelo.
        """
        batchsize = x.shape[0]

        # Capa 1
        x = self.linear_1(x)
        x = self.relu(x)
        
        # Capa LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Reajusta la salida de la celda oculta en [batch, features] para `linear_2`
        x = h_n.permute(1, 0, 2).reshape(batchsize, -1) 
        
        # Capa 2
        x = self.dropout(x)
        predictions = self.linear_2(x)
        return predictions[:, -1]  # Retorna solo la última predicción


def run_epoch(dataloader, is_training=False):
    """
    Ejecuta una época de entrenamiento o evaluación.

    Args:
        dataloader (DataLoader): DataLoader que contiene los datos de entrada y salida.
        is_training (bool, optional): Indica si es una época de entrenamiento. Por defecto es False.

    Returns:
        float: Pérdida promedio de la época.
        float: Tasa de aprendizaje actual.
    """
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            optimizer.zero_grad()

        batchsize = x.shape[0]

        x = x.to(config["training"]["device"])
        y = y.to(config["training"]["device"])

        out = model(x)
        loss = criterion(out.contiguous(), y.contiguous())

        if is_training:
            loss.backward()
            optimizer.step()

        # Accumulate epoch loss
        epoch_loss += (loss.detach().item() / batchsize)

        if is_training:
            print(f'Batch [{idx+1}/{len(dataloader)}], Loss: {loss.item():.6f}')
            if (idx+1) % 500 == 0:
                torch.save(model.state_dict(), f'model_batch_{idx+1}_2.pth')
                print(f'Model Saved')


    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr



#===========================================================
# Cragar base de entrenamiento
#===========================================================

dir_path = "mp3/canciones_train/"
sequences = []
for file_name in os.listdir(dir_path):
    if file_name.endswith(".mp3"):
        file_path = os.path.join(dir_path, file_name)
        waveform, sample_rate = load_audio(file_path, device)
        sequences.append(waveform)


# Concatenar todas las secuencias en una sola
all_sequences = torch.cat(sequences, dim=1)  # Concatenar en el eje de tiempo
del sequences

#===========================================================
# Entrenamiento de Modelo (Canal 1)
#===========================================================

model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

# Crear DataLoader para entrenamiento y validación

look_back = 40  # Longitud de la ventana de observación
batch_size =20000  # Tamaño del batch
data_loader = create_batches(all_sequences, look_back, batch_size,0)

# Definir el optimizador, el scheduler y la función de pérdida
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

# Iniciar el entrenamiento
for epoch in range(1):
    loss_train, lr_train = run_epoch(data_loader, is_training=True)
    #loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()
    
    print('Epoch[{}/{}] | loss train:{:.6f} | lr:{:.6f}'
              .format(epoch+1, config["training"]["num_epoch"], loss_train, lr_train))


#===========================================================
# Entrenamiento de Modelo (Canal 2)
#===========================================================

model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

# Crear DataLoader para entrenamiento y validación

look_back = 40  # Longitud de la ventana de observación
batch_size =20000  # Tamaño del batch
data_loader = create_batches(all_sequences, look_back, batch_size,0)

# Definir el optimizador, el scheduler y la función de pérdida
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

# Iniciar el entrenamiento
for epoch in range(1):
    loss_train, lr_train = run_epoch(data_loader, is_training=True)
    #loss_val, lr_val = run_epoch(val_dataloader)
    scheduler.step()
    
    print('Epoch[{}/{}] | loss train:{:.6f} | lr:{:.6f}'
              .format(epoch+1, config["training"]["num_epoch"], loss_train, lr_train))


#===========================================================
# Cargar datos de prueba (Canal 1)
#===========================================================

# Directorio con archivos MP3 de prueba 
dir_path = "mp3/canciones_prueba/"
sequences = []
for file_name in os.listdir(dir_path):
    if file_name.endswith(".mp3"):
        file_path = os.path.join(dir_path, file_name)
        waveform, sample_rate = load_audio(file_path, device)
        sequences.append(waveform)


# Concatenar todas las secuencias en una sola
all_sequences = torch.cat(sequences, dim=1)  # Concatenar en el eje de tiempo
del sequences


# Creaar batches
data_loader_val = create_batches(all_sequences, look_back, batch_size, 0)


#===========================================================
# Evaluación del modelo (Canal 1)
#===========================================================

model_1 = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])

# Mover el modelo al dispositivo adecuado (CPU o GPU)
device = config["training"]["device"]
model_1 = model_1.to(device)

# Cargar el estado del modelo desde el archivo guardado
checkpoint_path = 'model_batch_2000.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)  # Asegúrate de especificar map_location para moverlo al dispositivo correcto
model_1.load_state_dict(checkpoint)

# Poner el modelo en modo evaluación
model_1.eval()


predictions = []
true_values = []

with torch.no_grad():
    for batch in data_loader_val:
        inputs, targets = batch
        inputs = inputs.to(config["training"]["device"])
        targets = targets.to(config["training"]["device"])
        
        # Realizar predicciones
        outputs = model_1(inputs)
        
        # Guardar predicciones y valores verdaderos
        predictions.append(outputs.cpu().numpy())
        true_values.append(targets.cpu().numpy())

# Convertir listas a arrays numpy
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

# Evaluar el rendimiento
mse_loss = np.mean((predictions - true_values)**2)
print(f'MSE en el conjunto de prueba: {mse_loss:.6f}')

# MSE en el conjunto de prueba: 0.001354



#===========================================================
# Cargar datos de prueba (Canal 2)
#===========================================================

# Directorio con archivos MP3 de prueba 
dir_path = "mp3/canciones_prueba/"
sequences = []
for file_name in os.listdir(dir_path):
    if file_name.endswith(".mp3"):
        file_path = os.path.join(dir_path, file_name)
        waveform, sample_rate = load_audio(file_path, device)
        sequences.append(waveform)


# Concatenar todas las secuencias en una sola
all_sequences = torch.cat(sequences, dim=1)  # Concatenar en el eje de tiempo
del sequences


# Creaar batches
data_loader_val = create_batches(all_sequences, look_back, batch_size, 1)


#===========================================================
# Evaluación del modelo (Canal 2)
#===========================================================

model_2 = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"], num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])

# Mover el modelo al dispositivo adecuado (CPU o GPU)
device = config["training"]["device"]
model_2 = model.to(device)

# Cargar el estado del modelo desde el archivo guardado
checkpoint_path = 'model_batch_2000.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)  # Asegúrate de especificar map_location para moverlo al dispositivo correcto
model_2.load_state_dict(checkpoint)

# Poner el modelo en modo evaluación
model_2.eval()


predictions = []
true_values = []

with torch.no_grad():
    for batch in data_loader_val:
        inputs, targets = batch
        inputs = inputs.to(config["training"]["device"])
        targets = targets.to(config["training"]["device"])
        
        # Realizar predicciones
        outputs = model_2(inputs)
        
        # Guardar predicciones y valores verdaderos
        predictions.append(outputs.cpu().numpy())
        true_values.append(targets.cpu().numpy())

# Convertir listas a arrays numpy
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

# Evaluar el rendimiento
mse_loss = np.mean((predictions - true_values)**2)
print(f'MSE en el conjunto de prueba: {mse_loss:.6f}')

# MSE en el conjunto de prueba: 0.001395




























