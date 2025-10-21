import tensorflow as tf

def finetune_model(saved_model_path: str, dataset_path: str, epochs: int = 10):
    model = tf.saved_model.load(saved_model_path)
    
    # Freeze an arbritary amount of backbone layers for transfer learning
    for layer in model.layers[:-3]:
        layer.trainable = False
    
    train_ds = load_custom_dataset(dataset_path)
    
    # Compile and train
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='...')
    model.fit(train_ds, epochs=epochs)
    
    # Save fine-tuned model
    output_path = f"{saved_model_path}_finetuned"
    tf.saved_model.save(model, output_path)
    return output_path

def load_custom_dataset(dataset_path: str):
    pass