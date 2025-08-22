# ml_models.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import (Dense, Dropout, Input, LayerNormalization,
                                     MultiHeadAttention, TimeDistributed)
from tensorflow.keras.models import Model

# --- Helper function to create sequences from time series data ---
def create_sequences(data, seq_length):
    """Creates overlapping sequences from a time series array."""
    xs = []
    for i in range(len(data) - seq_length + 1):
        x = data[i:(i + seq_length)]
        xs.append(x)
    return np.array(xs)

# --- Transformer Layer Definitions ---
def transformer_encoder_layer(d_model, num_heads, dff, rate=0.1):
    """Creates a single Transformer encoder layer."""
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    ffn = tf.keras.Sequential([
        Dense(dff, activation='relu'),
        Dense(d_model)
    ])
    layernorm1 = LayerNormalization(epsilon=1e-6)
    layernorm2 = LayerNormalization(epsilon=1e-6)
    dropout1 = Dropout(rate)
    dropout2 = Dropout(rate)

    def call(x, training):
        attn_output = attn(x, x, x)
        attn_output = dropout1(attn_output, training=training)
        out1 = layernorm1(x + attn_output)

        ffn_output = ffn(out1)
        ffn_output = dropout2(ffn_output, training=training)
        out2 = layernorm2(out1 + ffn_output)
        return out2

    return call

def transformer_decoder_layer(d_model, num_heads, dff, rate=0.1):
    """Creates a single Transformer decoder layer."""
    attn1 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model) # Self-attention
    attn2 = MultiHeadAttention(num_heads=num_heads, key_dim=d_model) # Encoder-decoder attention
    ffn = tf.keras.Sequential([
        Dense(dff, activation='relu'),
        Dense(d_model)
    ])
    layernorm1 = LayerNormalization(epsilon=1e-6)
    layernorm2 = LayerNormalization(epsilon=1e-6)
    layernorm3 = LayerNormalization(epsilon=1e-6)
    dropout1 = Dropout(rate)
    dropout2 = Dropout(rate)
    dropout3 = Dropout(rate)

    def call(x, enc_output, training): # <- REMOVED look_ahead_mask
        # For an autoencoder, the self-attention in the decoder can see the whole sequence.
        # Thus, the mask is not needed.
        attn1_output = attn1(x, x, x) # <- REMOVED attention_mask
        attn1_output = dropout1(attn1_output, training=training)
        out1 = layernorm1(x + attn1_output)

        attn2_output = attn2(out1, enc_output, enc_output)
        attn2_output = dropout2(attn2_output, training=training)
        out2 = layernorm2(out1 + attn2_output)

        ffn_output = ffn(out2)
        ffn_output = dropout3(ffn_output, training=training)
        out3 = layernorm3(out2 + ffn_output)
        return out3

    return call

# --- Main TransformerAutoencoder Class ---
class TransformerAutoencoder:
    def __init__(self, seq_length, num_layers, d_model, num_heads, dff, rate=0.1):
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        self.threshold = 0.0
        self.model = self._build_model(num_layers, d_model, num_heads, dff, rate)

    def _build_model(self, num_layers, d_model, num_heads, dff, rate):
        """Builds and compiles the Keras model."""
        input_shape = (self.seq_length, 1)
        input_sequence = Input(shape=input_shape)

        # Encoder
        encoder_output = input_sequence
        for _ in range(num_layers):
            encoder_output = transformer_encoder_layer(d_model, num_heads, dff, rate)(encoder_output, training=True)

        # Decoder
        decoder_input = input_sequence
        decoder_output = decoder_input
        for _ in range(num_layers):
            # The decoder's call is now simpler without the mask
            decoder_output = transformer_decoder_layer(d_model, num_heads, dff, rate)(decoder_output, encoder_output, training=True)

        # Output layer
        output_sequence = TimeDistributed(Dense(1))(decoder_output)
        model = Model(inputs=input_sequence, outputs=output_sequence)
        model.compile(optimizer='adam', loss='mae') # MAE is often better for time series
        return model

    def train(self, df: pd.DataFrame):
        """Preprocesses data, trains the model, and sets the anomaly threshold."""
        if 'y' not in df.columns or len(df) < self.seq_length:
            print("Not enough data to train. Skipping.")
            return

        # Scale data
        data_scaled = self.scaler.fit_transform(df[['y']])
        
        # Create sequences
        sequences = create_sequences(data_scaled, self.seq_length)
        if len(sequences) == 0:
            print("Could not create any sequences from the data. Skipping training.")
            return

        # Train the model
        self.model.fit(sequences, sequences, epochs=20, batch_size=32, verbose=0)
        
        # Determine anomaly threshold from training reconstruction errors
        reconstructions = self.model.predict(sequences)
        train_loss = np.mean(np.abs(reconstructions - sequences), axis=(1, 2))
        
        # Set threshold using the 3-sigma rule (mean + 3 * standard deviation)
        self.threshold = np.mean(train_loss) + 3 * np.std(train_loss)
        print(f"Model training complete. Anomaly threshold set to: {self.threshold:.4f}")

    def predict_anomalies(self, df: pd.DataFrame):
        """Predicts anomalies and generates a forecast based on model reconstructions."""
        if 'y' not in df.columns or len(df) < self.seq_length:
            return {"forecast": [], "anomalies": []}

        data_scaled = self.scaler.transform(df[['y']])
        sequences = create_sequences(data_scaled, self.seq_length)
        
        if len(sequences) == 0:
            return {"forecast": [], "anomalies": []}

        reconstructions_scaled = self.model.predict(sequences)
        loss_per_sequence = np.mean(np.abs(reconstructions_scaled - sequences), axis=(1, 2))
        
        # Prepare a dataframe to hold results
        full_df = df.copy()
        full_df['anomaly'] = False
        full_df['loss'] = 0.0

        # Attribute the reconstruction loss of a sequence to its final data point
        for i, loss in enumerate(loss_per_sequence):
            idx = i + self.seq_length - 1
            full_df.loc[idx, 'loss'] = loss
            if loss > self.threshold:
                full_df.loc[idx, 'anomaly'] = True

        anomalies = full_df[full_df['anomaly'] == True]
        
        # Generate the "forecast" (the model's reconstruction of what the data should have been)
        # Pad the start since reconstructions begin only after the first sequence is complete
        reconstructed_values_scaled = np.vstack([
            np.full((self.seq_length - 1, 1), np.nan),
            reconstructions_scaled[:, -1, :]
        ])
        reconstructed_values = self.scaler.inverse_transform(reconstructed_values_scaled)
        
        full_df['forecast'] = reconstructed_values.flatten()
        
        # Format output for JSON response
        forecast_output = [
            {"date": row['ds'].strftime('%Y-%m-%d'), "cost": row['y'], "forecast_cost": row['forecast']}
            for _, row in full_df.iterrows()
        ]

        anomalies_output = [
            {"date": row['ds'].strftime('%Y-%m-%d'), "cost": row['y'], "reconstruction_error": row['loss']}
            for _, row in anomalies.iterrows()
        ]

        return {"forecast": forecast_output, "anomalies": anomalies_output}