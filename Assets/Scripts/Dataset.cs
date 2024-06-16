using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

public class Dataset : MonoBehaviour
{
    // Utilities for loading and handling datasets 
    
    public static Dataset dataset;
    
    public List<int[]> imageDataList = new List<int[]>();
    public List<int> imageDataLabels = new List<int>();

    private int inputImageSize = 28;
    private int targetImageSize = 24;
        
    static Unity.Mathematics.Random _rand;

    private void Awake()
    {
        dataset = this;
        _rand = new Unity.Mathematics.Random(123);
    }

    public (float[,], float[,]) LoadMultiple(string[] filepaths, int[] labelOffsets, bool returnData)
    {
        // Load multiple datasets from filepaths and apply transformations and binarization to the images
        
        List<int[]> imageDataListTemp = new List<int[]>();
        List<int> imageDataLabelsTemp = new List<int>();
        
        if (filepaths.Length != labelOffsets.Length)
        {
            throw new ArgumentException("The lengths of filepaths and labelOffsets must be the same.");
        }

        for (int i = 0; i < filepaths.Length; i++)
        {
            string currentFilePath = filepaths[i];
            int currentLabelOffset = labelOffsets[i];

            (List<int[]> currentImageDataList, List<int> currentImageDataLabels) =
                ReadCsv(currentFilePath, currentLabelOffset);

            for (int j = 0; j < currentImageDataList.Count; j++)
            {
                currentImageDataList[j] = ImageTransformations.Transformations.ProcessImage(currentImageDataList[j],
                    inputImageSize, inputImageSize, targetImageSize, targetImageSize);
                currentImageDataList[j] =
                    ImageTransformations.Transformations.BinarizeImage(currentImageDataList[j], 50);
            }

            imageDataListTemp.AddRange(currentImageDataList);
            imageDataLabelsTemp.AddRange(currentImageDataLabels);
        }

        if (returnData)
        {
            float[,] Inputs = new float[imageDataListTemp.Count, imageDataListTemp[0].Length];
            float[,] Labels = new float[imageDataListTemp.Count, 1];

            for (int i = 0; i < imageDataListTemp.Count; i++)
            {
                for (int j = 0; j < imageDataListTemp[i].Length; j++)
                {
                    Inputs[i, j] = imageDataListTemp[i][j];
                }

                Labels[i, 0] = imageDataLabelsTemp[i];
            }

            return (Inputs, Labels);
        }
        else
        {
            imageDataList.AddRange(imageDataListTemp);
            imageDataLabels.AddRange(imageDataLabelsTemp);

            
            var occurrences = imageDataLabels
                .GroupBy(n => n)
                .Select(g => new { Number = g.Key, Count = g.Count() });

            foreach (var item in occurrences)
            {
                Debug.Log($"Number: {item.Number}, Occurrences: {item.Count}");
            }
            
            return (new float[0, 0], new float[0, 0]);
        }
        
    }

    public (List<int[]>, List<int>) ReadCsv(string path, int labelOffset, int sizeLimit = 20000000)
    {
        // Read image data and labels from a CSV file, applying a label offset in case of different types of data sources

        List<int[]> dataImages = new List<int[]>();
        List<int> dataLabels = new List<int>();

        using (StreamReader reader = new StreamReader(path))
        {
            int linesRead = 0;
            while (!reader.EndOfStream && linesRead < sizeLimit)
            {
                string line = reader.ReadLine();
                if (line == null) break;

                string[] values = line.Split(',');
                if (int.TryParse(values[0], out int label))
                {
                    dataLabels.Add(label + labelOffset);

                    int[] imageData = new int[values.Length - 1];
                    Parallel.For(0, imageData.Length, i =>
                    {
                        int.TryParse(values[i + 1], out imageData[i]);
                    });
                    dataImages.Add(imageData);
                    linesRead++;
                }
            }
        }
        return (dataImages, dataLabels);
    }

    public (int[][] trainData, int[] trainLabels, int[][] validationData, int[] validationLabels, int[][] testData, int[] testLabels) SplitData(float trainRatio, float validationRatio)
    {
        // Split the dataset into training, validation, and test sets
        
        if (trainRatio + validationRatio > 1.0f)
        {
            throw new ArgumentException("The sum of trainRatio and validationRatio must be less than or equal to 1.");
        }

        int dataSize = imageDataList.Count;
        int[] indices = Enumerable.Range(0, dataSize).ToArray();
        indices = indices.OrderBy(i => _rand.NextInt()).ToArray();

        int trainSize = (int)(dataSize * trainRatio);
        int validationSize = (int)(dataSize * validationRatio);
        int testSize = dataSize - trainSize - validationSize;

        int[][] trainData = new int[trainSize][];
        int[] trainLabels = new int[trainSize];
        int[][] validationData = new int[validationSize][];
        int[] validationLabels = new int[validationSize];
        int[][] testData = new int[testSize][];
        int[] testLabels = new int[testSize];

        for (int i = 0; i < dataSize; i++)
        {
            int[] data = imageDataList[indices[i]];
            int label = imageDataLabels[indices[i]];

            if (i < trainSize)
            {
                trainData[i] = data;
                trainLabels[i] = label;
            }
            else if (i < trainSize + validationSize)
            {
                validationData[i - trainSize] = data;
                validationLabels[i - trainSize] = label;
            }
            else
            {
                testData[i - trainSize - validationSize] = data;
                testLabels[i - trainSize - validationSize] = label;
            }
        }

        return (trainData, trainLabels, validationData, validationLabels, testData, testLabels);
    }

    public void AugmentData(ref int[][] data, ref int[] labels, List<int> minorityClasses)
    {
        // Augment the data by rotating images of minority classes (2 for each instance)

        var augmentedData = new List<int[]>(data);
        var augmentedLabels = new List<int>(labels);

        for (int i = 0; i < data.Length; i++)
        {
            if (minorityClasses.Contains(labels[i]))
            {
                int[] rotatedData1 = ImageTransformations.Transformations.RotateImage(data[i], targetImageSize, targetImageSize, _rand.NextInt(-15, 5));
                int[] rotatedData2 = ImageTransformations.Transformations.RotateImage(data[i], targetImageSize, targetImageSize, _rand.NextInt(5, 15));

                augmentedData.Add(rotatedData1);
                augmentedLabels.Add(labels[i]);
                augmentedData.Add(rotatedData2);
                augmentedLabels.Add(labels[i]);
            }
        }

        data = augmentedData.ToArray();
        labels = augmentedLabels.ToArray();
    }


    public void ShuffleData(int[][] trainData, int[] trainLabels)
    {
        // Shuffle the training data and labels
        
        for (int i = 0; i < trainData.Length - 1; i++)
        {
            int randomIndex = _rand.NextInt(i, trainData.Length);
            (trainData[i], trainData[randomIndex]) = (trainData[randomIndex], trainData[i]);
            (trainLabels[i], trainLabels[randomIndex]) = (trainLabels[randomIndex], trainLabels[i]);
        }
    }
}
