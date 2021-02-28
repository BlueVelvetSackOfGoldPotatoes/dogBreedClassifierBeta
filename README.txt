# Tried models
MobileNetV2 - accuracy around 70% after 30 epochs (with clear stagnation of learning around 70%)

# Random dogs from the internet
    '''
    get_name = info.features['label'].int2str
    decode = lambda x: get_name(tf.math.argmax(x))

    filename_dataset = tf.data.Dataset.list_files("Doggos/*.jp*g")
    image_dataset = filename_dataset.map(lambda x: tf.io.decode_jpeg(tf.io.read_file(x)))

    for dog in image_dataset:
        mock_dict = {'image': dog, 'label': 0}
        pic, _ = preprocess(mock_dict)

        plt.figure()
        plt.imshow(pic)

        img_tensor = tf.expand_dims(pic,0)
        pred = model(img_tensor)
        
        top_components = tf.reshape(tf.math.top_k(pred, k=5).indices,shape=[-1])
        top_matches = [get_name(i) for i in top_components]

        plt.title(top_matches[0])
        print(top_matches)
    '''
    
    main.py is a standalone program
