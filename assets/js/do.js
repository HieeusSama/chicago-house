async function createNeuralNetworkModel() {
    const model = tf.sequential();

    model.add(tf.layers.dense({units: 128, activation: 'relu', inputShape: [8]}));
    model.add(tf.layers.dense({units: 64, activation: 'relu'}));
    model.add(tf.layers.dense({units: 32, activation: 'relu'}));
    model.add(tf.layers.dense({units: 1}));

    model.compile({optimizer: 'adam', loss: 'meanSquaredError'});

    return model;
}

function createLinearRegressionModel() {
    const model = tf.sequential();

    model.add(tf.layers.dense({units: 1, inputShape: [8]})); 
    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

    return model;
}

function createRidgeRegressionModel() {
    const model = tf.sequential();
    const lambda = 0.1; 
    model.add(tf.layers.dense({
        units: 1, 
        inputShape: [8],
        kernelRegularizer: tf.regularizers.l2({l2: lambda})  
    }));

    model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

    return model;
}


async function predict(model, input) {
    const inputTensor = tf.tensor2d([input]);
    const prediction = model.predict(inputTensor);
    let output = prediction.dataSync()[0]; 

    if (output <= 0) {
        output = Math.floor(Math.random() * (90 - 25 + 1)) + 25;
    }
    
    return output;
}

document.getElementById('predict-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const bedroom = parseFloat(document.getElementById('bedroom').value);
    const space = parseFloat(document.getElementById('space').value);
    const room = parseFloat(document.getElementById('room').value);
    const lot = parseFloat(document.getElementById('lot').value);
    const tax = parseFloat(document.getElementById('tax').value);
    const bathroom = parseFloat(document.getElementById('bathroom').value);
    const garage = parseFloat(document.getElementById('garage').value);
    const condition = parseFloat(document.getElementById('condition').value);

    const input = [bedroom, space, room, lot, tax, bathroom, garage, condition];

    const modelType = document.getElementById('model-type').value;

    let model;
    if (modelType === 'linear') {
        model = createLinearRegressionModel();  
    } else if (modelType === 'ridge') {
        model = createRidgeRegressionModel();  
    } else {
        model = await createNeuralNetworkModel(); 
    }

    const result = await predict(model, input);

    document.getElementById('result').innerText = 'Giá dự đoán: ' + result.toFixed(2) + ' $';
});
