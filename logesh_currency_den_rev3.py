  import tensorflow as tf
  from tensorflow.keras.preprocessing import image
  from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
  from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
  from tensorflow.keras.models import Model
  from tensorflow.keras.optimizers import Adam
  target_size =(224,224)
  loaded_model = tf.keras.models.load_model('/content/drive/MyDrive/your_model_name.h5')

  few_shot_image_path = '/content/3f914916-837c-4160-b7d0-6a861b4afe94.jpg'
  few_shot_image = image.load_img(few_shot_image_path, target_size=target_size)
  few_shot_image_array = image.img_to_array(few_shot_image)
  few_shot_image_array = preprocess_input(few_shot_image_array)
  few_shot_image_array = tf.expand_dims(few_shot_image_array, 0)  # Add batch dimension


  prediction = loaded_model.predict(few_shot_image_array)
  # Get the denomination based on the highest predicted class
  denominations = ['1Hundrednote', '2Hundrednote', '2Thousandnote', '5Hundrednote', 'Fiftynote', 'Tennote', 'Twentynote']
  predicted_denomination = denominations[prediction.argmax()]
  print("Predicted Denomination:", predicted_denomination)
