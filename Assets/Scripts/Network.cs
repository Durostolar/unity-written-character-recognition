using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using Random = Unity.Mathematics.Random;

[System.Serializable]
public class LayerData
{
    public int nInputs;
    public int nNeurons;
    public float[] weights;
    public float[] biases;
    public Activation activation;
}

public class NetworkData
{
    public List<LayerData> layers = new();
}

public enum Activation
{
    Relu, Sigmoid
}

public static class ActivationFunctions
{
    public static float Relu(float x)
    {
        return x < 0 ? 0 : x;
    }

    public static float ReluDerivative(float x)
    {
        return x <= 0 ? 0 : 1;
    }

    public static float Sigmoid(float x)
    {
        return 1.0f / (1.0f + Mathf.Exp(-x));
    }

    public static float SigmoidDerivative(float output)
    {
        return output * (1 - output);
    }
}

[Serializable]
public class Network : MonoBehaviour
{
    // Class representing a neural network, managing layers, activations, and training procedures.

    public List<Layer> Layers = new();
    public int nCLasses;
    public float lr;

    public delegate float ActivationFunction(float x);

    public delegate float ActivationFunctionDerivative(float x);

    static Random _rand;

    public void Start()
    {
        _rand = new Random(123);
    }

    public class Layer
    {    
        // Class represents a single layer in the neural network, holding neurons, weights, biases, and related computations
        
        public readonly int NInputs;
        public readonly int NNeurons;

        public float[,] Weights;
        public float[] Biases;

        public float[,] Outputs;

        private float[,] nodeErrors;
        public readonly Activation Activation;

        private ActivationFunction activationFunction;
        private ActivationFunctionDerivative activationFunctionDerivative;

        public Layer(int nInputs, int nNeurons, Activation activation)
        {
            NInputs = nInputs;
            NNeurons = nNeurons;
            Activation = activation;
            SetActivations(activation);

            Biases = new float[nNeurons];

            Weights = new float[nNeurons, nInputs];

            InitializeHe();
        }

        private void SetActivations(Activation activation)
        {
            switch (activation)
            {
                case Activation.Relu:
                    activationFunction = ActivationFunctions.Relu;
                    activationFunctionDerivative = ActivationFunctions.ReluDerivative;
                    break;
                case Activation.Sigmoid:
                    activationFunction = ActivationFunctions.Sigmoid;
                    activationFunctionDerivative = ActivationFunctions.SigmoidDerivative;
                    break;
                default:
                    throw new ArgumentException("Unsupported activation type: {activation}");
            }
        }

        private void InitializeHe()
        {    
            // Initialize layer weights and biases using He initialization method
            
            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                Biases[i] = RandomInNormalDistribution() * Mathf.Sqrt(2f / NInputs);
                for (int j = 0; j < Weights.GetLength(1); j++)
                {
                    Weights[i, j] = RandomInNormalDistribution() * Mathf.Sqrt(2f / NInputs);
                }
            }

            float RandomInNormalDistribution()
            {
                float x1 = 1 - _rand.NextFloat();
                float x2 = 1 - _rand.NextFloat();
                return Mathf.Sqrt((float)(-2.0 * Mathf.Log(x1))) * Mathf.Cos((float)(2.0 * Mathf.PI * x2));
            }
        }

        public virtual void CalculateOutputs(float[,] inputs)
        { 
            // Compute the outputs for this layer given the inputs, i.e. forward pass
            
            LazyAllocate(ref Outputs, inputs.GetLength(0), NNeurons);

            Parallel.For(0, inputs.GetLength(0), b =>
            {
                for (int j = 0; j < NNeurons; j++)
                {
                    float sum = Biases[j];
                    for (int i = 0; i < NInputs; i++)
                    {
                        sum += Weights[j, i] * inputs[b, i];
                    }

                    Outputs[b, j] = activationFunction(sum);
                }
            });
        }

        private void LazyAllocate(ref float[,] matrix, int dim1, int dim2)
        {    
            // Ensure that matrix has the correct dimensions, reallocating if necessary.
            // Useful to save memory realocations when batch sizes are different
            
            if (matrix == null || matrix.GetLength(0) != dim1 || matrix.GetLength(1) != dim2)
            {
                matrix = new float[dim1, dim2];
            }
        }

        public void BackpropagateOuter(float[,] y, float[,] previousOutputs, float lr)
        {    
            // Backpropagation for the output layer
            
            LazyAllocate(ref nodeErrors, Outputs.GetLength(0), NNeurons);

            Parallel.For(0, Outputs.GetLength(0), b =>
            {
                for (int j = 0; j < NNeurons; j++)
                {
                    nodeErrors[b, j] = Outputs[b, j];
                    if (j == (int)y[b, 0]) nodeErrors[b, j] -= 1;
                }
            });
            UpdateWeights(lr, previousOutputs);
        }

        public virtual void BackpropagateInner(Layer nextLayer, float[,] previousOutputs, float lr)
        {    
            // Backpropagation for the input layer
            
            LazyAllocate(ref nodeErrors, Outputs.GetLength(0), NNeurons);
            
            for (int b = 0; b < Outputs.GetLength(0); b++)
            {
                for (int j = 0; j < NNeurons; j++)
                {
                    nodeErrors[b,j] = 0;
                    for (int r = 0; r < nextLayer.NNeurons; r++)
                    {
                        nodeErrors[b,j] += nextLayer.nodeErrors[b,r] * nextLayer.activationFunctionDerivative(nextLayer.Outputs[b,r]) *
                                           nextLayer.Weights[r, j];
                    }
                }
            }
            UpdateWeights(lr, previousOutputs);
        }

        public virtual void UpdateWeights(float lr, float[,] previousOutputs)
        {    
            // Update the weights and biases of the layer based on the computed gradients and learning rate
            
            float[,] derivatives = new float[Outputs.GetLength(0), NNeurons];

            for (int j = 0; j < NNeurons; j++)
            {
                for (int b = 0; b < Outputs.GetLength(0); b++)
                {
                    derivatives[b, j] = nodeErrors[b, j] * activationFunctionDerivative(Outputs[b, j]);
                }
            }

            Parallel.For(0, NNeurons, j =>
            {
                float batchError;
                for (int i = 0; i < NInputs; i++)
                {
                    batchError = 0;
                    for (int b = 0; b < Outputs.GetLength(0); b++)
                    {
                        batchError += derivatives[b, j] * previousOutputs[b, i];
                    }

                    Weights[j, i] -= lr * batchError;
                }

                batchError = 0;
                for (int b = 0; b < Outputs.GetLength(0); b++)
                {
                    batchError += derivatives[b, j];
                }

                Biases[j] -= lr * batchError;
            });
        }
    }


    public void AddLayer(int nInputs, int nNeurons, Activation activationType)
    {
        // Add a new layer with specified parameters

        Layer newLayer = new Layer(nInputs, nNeurons, activationType);
        Layers.Add(newLayer);
    }

    public void ClearLayers()
    {
        // Clear all layers
        
        Layers.Clear();
    }
    
    public void ForwardPass(float [,] input)
    {    
        // Conduct a forward pass through the network using the provided input matrix
        
        Layers[0].CalculateOutputs(input);
        for (int i = 1; i < Layers.Count; i++)
        {
            Layers[i].CalculateOutputs(Layers[i-1].Outputs);    
        }
    }
    
    public void BackwardPass(float [,] input, float [,] target)
    {     
        // Conduct a backward pass for training the network using input and target matrices
    
        Layers[^1].BackpropagateOuter(target,  Layers[^2].Outputs, lr);
        for (int i = Layers.Count-2; i > 0; i--)
        {
            Layers[i].BackpropagateInner(Layers[i+1], Layers[i-1].Outputs, lr);    
        }
        Layers[0].BackpropagateInner(Layers[1], input, lr);
    }
    
    public NetworkData GetData()
    {
        // Extract and return the current state of the network
        
        NetworkData data = new NetworkData();
        foreach (Layer layer in Layers)
        {
            LayerData layerData = new LayerData();
            layerData.nInputs = layer.NInputs;
            layerData.nNeurons = layer.NNeurons;
            layerData.weights = ImageTransformations.Transformations.Convert2DArrayTo1D(layer.Weights);
            layerData.biases = layer.Biases;
            layerData.activation = layer.Activation;
            data.layers.Add(layerData);
        }
        return data;
    }
    
    public void LoadData(NetworkData data)
    {    
        // Load network configuration and parameters from a given data structure
        
        Layers.Clear();
        foreach (LayerData layerData in data.layers)
        {
            Layer layer = new Layer(layerData.nInputs, layerData.nNeurons, layerData.activation);
            layer.Weights = ImageTransformations.Transformations.Convert1DArrayTo2D(layerData.weights, layerData.nNeurons, layerData.nInputs);
            layer.Biases = layerData.biases;
            Layers.Add(layer);
            nCLasses = Layers[^1].NNeurons;
        }
    }
    
    public void SaveNetParams()
    {    
        // Saves the network's parameters and configuration to a JSON file
        
        try
        {
            NetworkData data = GetData();
            string json = JsonUtility.ToJson(data);
            File.WriteAllText("Assets/Resources/model.json", json);
        }
        catch (IOException ex)
        {
            Debug.LogError($"Failed to save network parameters: {ex.Message}");
        }
    }
    
    public double CalculateLoss(float[,] labels)
    {
        // Calculate and log the mean squared error loss
        
        double mse = 0;

        for (int i = 0; i < labels.GetLength(0); i++)
        {
            int trueLabel = (int)labels[i, 0];
            
            for (int j = 0; j < Layers[1].NNeurons; j++)
            {
                if (j == trueLabel)
                {
                    mse += Math.Pow(Layers[1].Outputs[i, j] - 1,2);
                }
                else
                {
                    mse += Math.Pow(Layers[1].Outputs[i, j],2);
                }
            }
        }
        mse /= labels.GetLength(0);
        Debug.Log("MSE: " + mse);
        return mse;
    }
    
    public void Conf(float[,] labels)
    {    
        // Construct and log a confusion matrix and calculate the classification accuracy
        
        int hit = 0;
        int[,] confusionMatrix = new int[nCLasses,nCLasses];
        
        for (int i = 0; i < labels.GetLength(0); i++)
        {
            int maxIndex = 0;
            int trueLabel = (int)labels[i, 0];
            for (int j = 1; j < Layers[1].NNeurons; j++)
            {
                if (Layers[1].Outputs[i, j] >= Layers[1].Outputs[i,maxIndex])
                {
                    maxIndex = j;
                }
            }
            confusionMatrix[trueLabel, maxIndex]++;
            if (maxIndex == trueLabel)
            {
                hit++;
            }
        }

        string accuracies = "";

        for (int i = 0; i < nCLasses; i++)
        {
            for (int j = 0; j < nCLasses; j++)
            {
                accuracies+=confusionMatrix[i,j]+",";
            }
            Debug.Log(accuracies);
            accuracies = "";
        }
        Debug.Log("Total Accuracy " + hit + " from " + labels.GetLength(0));
    }

    public class BestModelCallback
    {    
        // Handles the logic for tracking and responding to the best model based on validation loss
        
        public double BestValidLoss { get; private set; } = double.MaxValue;
        private Action onNewBest;

        public BestModelCallback(Action onNewBest)
        {
            this.onNewBest = onNewBest;
        }

        public void Invoke(double validLoss)
        {
            if (validLoss < BestValidLoss)
            {
                BestValidLoss = validLoss;
                Debug.Log("Saving the model");
                onNewBest?.Invoke(); 
            }
        }
    }
}