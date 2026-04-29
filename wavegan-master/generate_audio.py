import os
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write

# ==============================
# CONFIG
# ==============================
MODEL_DIR = "./train_drone"   # ou train_noise
OUTPUT_DIR = "./generated_audio"
NUM_SAMPLES = 50
SAMPLE_RATE = 16000

# ==============================
# SETUP
# ==============================
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

tf.reset_default_graph()

# ==============================
# LOAD MODEL
# ==============================
saver = tf.train.import_meta_graph(os.path.join(MODEL_DIR, 'infer.meta'))
graph = tf.get_default_graph()

sess = tf.InteractiveSession()
saver.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))

# ==============================
# GET TENSORS
# ==============================
z = graph.get_tensor_by_name('z:0')
G_z = graph.get_tensor_by_name('G_z:0')

# ==============================
# GENERATE AUDIO
# ==============================
print("Gerando áudios...")

for i in range(NUM_SAMPLES):
    # vetor latente aleatório
    _z = (np.random.rand(1, 100) * 2.) - 1
    
    # gerar áudio
    audio = sess.run(G_z, {z: _z})
    
    # pegar canal 0
    audio = audio[0, :, 0]
    
    # normalizar
    audio = audio / np.max(np.abs(audio))
    
    # salvar
    filename = os.path.join(OUTPUT_DIR, f"sample_{i}.wav")
    write(filename, SAMPLE_RATE, audio.astype(np.float32))
    
    print(f"Salvo: {filename}")

print("Finalizado!")