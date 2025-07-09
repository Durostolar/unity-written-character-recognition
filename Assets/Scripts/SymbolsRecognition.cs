using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SymbolsRecognition : MonoBehaviour
{
    // This class manages the operations for training network for a specific task - to recognize symbols
    // It initializes network settings, loads data, performs training, and handles data preprocessing and augmentation
        
    [SerializeField] Network network;
    
    private int nClasses = 36;
    
    private float learningRate = 0.001f;
    private int epochs = 5;
    private int batchSize = 32;
    
    private void InitializeNetworkForMnist()
    {
        // Initialize the network with specific settings for the task
        
        network.AddLayer(24*24,200,Activation.Relu);
        network.AddLayer(200,nClasses,Activation.Sigmoid);

        network.nCLasses = nClasses;
        network.lr = learningRate;
    }

    private readonly string[] csvFilePaths = 
    {
        "Assets/Data/Numbers/mnist_train.csv",
        "Assets/Data/Numbers/mnist_test.csv",
        "Assets/Data/AZcsv/data.csv"
    };
    
    public void LoadData()
    {
        Dataset.dataset.LoadMultiple(csvFilePaths, new int[] {0,0,10}, false);
    }
    
    public void Train()
    {
        // Train the model, Log losses during training, evaluate performance on test split at the end
        
        InitializeNetworkForMnist();
        
        var (trainData, trainLabels, validationData, validationLabels, testData, testLabels) = Dataset.dataset.SplitData(0.8f,0.1f);
        
        // Augumentation
        List<int> minorityClasses = new List<int> {11, 15, 18, 26 }; // Minority classes that need to be addressed (empirically created)
        Dataset.dataset.AugmentData(ref trainData, ref trainLabels, minorityClasses);
        Dataset.dataset.AugmentData(ref validationData, ref validationLabels, minorityClasses);
        
        // Prepare the validation and test data
        (float[,] validationDataInputs, float[,] validationDataLabels) = ConvertTo2DArray(validationData, validationLabels);
        (float[,] testDataInputs, float[,] testDataLabels) = ConvertTo2DArray(testData, testLabels);
        
        network.ForwardPass(validationDataInputs);
        network.CalculateLoss(validationDataLabels);
    
        Network.BestModelCallback callback = new Network.BestModelCallback(network.SaveNetParams);
    
        float[,] batchInputs = new float[1,1];
        float[,] batchTargets = new float[1,1];
    
        // Training cycle
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            Dataset.dataset.ShuffleData(trainData, trainLabels);
    
            // Process data in batches
            for (int i = 0; i < trainData.Length; i += batchSize)
            {
                int batchSizeActual = Mathf.Min(batchSize, trainData.Length - i);
    
                if (batchInputs.GetLength(0) != batchSizeActual)
                {
                    batchInputs = new float[batchSizeActual, trainData[i].Length];
                    batchTargets = new float[batchSizeActual, 1];
                }
    
                for (int j = 0; j < batchSizeActual; j++)
                {
                    for (int k = 0; k < trainData[0].Length; k++)
                    {
                        batchInputs[j, k] = trainData[i + j][k];
                    }
    
                    batchTargets[j, 0] = trainLabels[i + j];
                }
    
                network.ForwardPass(batchInputs);
                network.BackwardPass(batchInputs, batchTargets);
            }
    
            Debug.Log("After Epoch " + epoch);
            network.ForwardPass(validationDataInputs);
            double loss = network.CalculateLoss(validationDataLabels);
    
             //Callback, skip it in the first epochs to save time
            if (epoch > 15)
            {
                callback.Invoke(loss);
            }
        }
        EvaulateTestSet(testDataInputs, testDataLabels, network);
    }
    
    public void EvaluateCollectedTest(Network net)
    {
        // Evaluate on an additional test set from an external file, can be used with other than default network

        string[] testFilePaths = { "Assets/Data/collectedData.csv"};
        int[] offset = {0};

        (float[,] testInputs, float[,] testLabels) = Dataset.dataset.LoadMultiple(testFilePaths, offset, true);

        if (net == null)
        {
            EvaulateTestSet(testInputs, testLabels, network);
        }
        else
        {
            EvaulateTestSet(testInputs, testLabels, net);
        }
    }
    
    private double EvaulateTestSet(float[,] testSet, float[,] testLabels, Network net)
    {
        // Calculate metrics on an test set
    
        net.ForwardPass(testSet);
        Debug.Log("Testing results:");
        double loss =net.CalculateLoss(testLabels);
        net.Conf(testLabels);
        return loss;
    }
    
    public static (float[,], float[,]) ConvertTo2DArray(int[][] data, int[] labels)
    {
        // Aux. method to convert data into suitable format
        
        int rows = data.GetLength(0);
        int cols = data[0].Length;

        float[,] dataInputs = new float[rows, cols];
        float[,] dataLabels = new float[rows, 1];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                dataInputs[i, j] = data[i][j];
            }
            dataLabels[i, 0] = labels[i];
        }

        return (dataInputs, dataLabels);
    }
}
