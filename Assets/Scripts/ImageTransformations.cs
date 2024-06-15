using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ImageTransformations : MonoBehaviour
{
    // This class provides a set of image processing methods
    
    public static ImageTransformations Transformations;
    
    private void Awake()
    {
        Transformations = this;
    }
    
    public int[] ProcessImage(int[] rawImage, int width, int height, int targetWidth, int targetHeight)
    {
        // Processes the image by finding the bounding box, centralize and resize it
        
        var (minX, minY, maxX, maxY) = FindBoundingBox(rawImage, width,  height);
        int[] processedImage = CentralizeAndResize(rawImage, width, height, targetWidth, targetHeight, minX, minY, maxX, maxY);
        return processedImage;
    }
    
    private (int, int, int, int) FindBoundingBox(int[] image, int width, int height)
    {
        // Finds the bounding box of non-zero pixels in the image
        
        int minX = width, maxX = -1, minY = height, maxY = -1;
        for (int i = 0; i < image.Length; i++)
        {
            int x = i % width;
            int y = i / width;
            if (image[i] != 0)
            {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
        if (minX > maxX) return (0, 0, width - 1, height - 1);
        return (minX, minY, maxX, maxY);
    }
    
    private int[] CentralizeAndResize(int[] image, int width, int height, int targetWidth, int targetHeight, int minX, int minY, int maxX, int maxY)
    {
        // Centralize the image and resize it to the target dimensions
        
        int[] result = new int[targetWidth * targetHeight];
        float bboxWidth = maxX - minX + 1;
        float bboxHeight = maxY - minY + 1;
        float bboxCenterX = minX + bboxWidth / 2f;
        float bboxCenterY = minY + bboxHeight / 2f;
        float targetCenterX = targetWidth / 2f;
        float targetCenterY = targetHeight / 2f;

        float scale = Mathf.Min(targetWidth / bboxWidth, targetHeight / bboxHeight);

        for (int y = 0; y < targetHeight; y++)
        {
            for (int x = 0; x < targetWidth; x++)
            {
                // Calculate corresponding pixel in the original image
                int originalX = Mathf.FloorToInt((x - targetCenterX) / scale + bboxCenterX);
                int originalY = Mathf.FloorToInt((y - targetCenterY) / scale + bboxCenterY);
                if (originalX >= 0 && originalX < width && originalY >= 0 && originalY < height)
                {
                    result[y * targetWidth + x] = image[originalY * width + originalX];
                }
                else
                {
                    result[y * targetWidth + x] = 0;  // Fill out-of-bound areas with zero
                }
            }
        }
        return result;
    }

    public int[] BinarizeImage(int[] input, int threshold)
    {
        // Transform pixel values to 0/1 based on a threshold
        
        int[] result = new int[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            result[i] = input[i] > threshold ? 1 : 0;
        }

        return result;
    }
    
    public void BinarizeData(int[][] dataSet, int threshold=50)
    {
        // Binarize all images in the dataset
        
        for (int i = 0; i < dataSet.GetLength(0); i++)
        {
            dataSet[i] = BinarizeImage(dataSet[i], threshold); 
        }
    }
    
    public int[] RotateImage(int[] imageData, int width, int height, float angle)
    {
        // Rotate a 1D image array by a given angle (in degrees)
        
        int[,] matrix = Convert1DArrayTo2D(imageData, width, height);
        int[,] rotatedMatrix = RotateMatrix(matrix, width, height, angle);
        return Convert2DArrayTo1D(rotatedMatrix);
    }
    
    public T[,] Convert1DArrayTo2D<T>(T[] array, int rowCount, int colCount)
    {
        // Convert 1D array to a 2D array
        
        if (array.Length != rowCount * colCount)
            throw new ArgumentException("The total size of the new array must be equal to the size of the old array.");

        T[,] reshapedArray = new T[rowCount, colCount];
        for (int i = 0; i < rowCount; i++)
        {
            for (int j = 0; j < colCount; j++)
            {
                reshapedArray[i, j] = array[i * colCount + j];
            }
        }
        return reshapedArray;
    }

    public T[] Convert2DArrayTo1D<T>(T[,] array)
    {
        // Convert 2D matrix to 1D array
        
        int rowCount = array.GetLength(0);
        int colCount = array.GetLength(1);
        T[] flattenedArray = new T[rowCount * colCount];
        for (int i = 0; i < rowCount; i++)
        {
            for (int j = 0; j < colCount; j++)
            {
                flattenedArray[i * colCount + j] = array[i, j];
            }
        }
        return flattenedArray;
    }
    
    private int[,] RotateMatrix(int[,] matrix, int width, int height, float angle)
    {        
        // Rotate matrix by a given angle
        
        int[,] rotatedMatrix = new int[height, width];
        double radAngle = angle * (Math.PI / 180);
        double cosAngle = Math.Cos(radAngle);
        double sinAngle = Math.Sin(radAngle);
        int centerX = width / 2;
        int centerY = height / 2;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int originalX = (int)(cosAngle * (x - centerX) - sinAngle * (y - centerY) + centerX);
                int originalY = (int)(sinAngle * (x - centerX) + cosAngle * (y - centerY) + centerY);
                if (originalX >= 0 && originalX < width && originalY >= 0 && originalY < height)
                {
                    rotatedMatrix[y, x] = matrix[originalY, originalX];
                }
                else
                {
                    rotatedMatrix[y, x] = 0; // Assign zero if the original coordinates are out of bounds
                }
            }
        }
        return rotatedMatrix;
    }
    
    public Texture2D ScaleTexture(Texture2D source, int targetWidth, int targetHeight)
    {
        // Scale texture to the target width and height
        
        Texture2D result = new Texture2D(targetWidth, targetHeight, source.format, true);
        Color[] rpixels = result.GetPixels(0);
        float incX = (1.0f / source.width) * ((float)source.width / targetWidth);
        float incY = (1.0f / source.height) * ((float)source.height / targetHeight);
        for (int px = 0; px < rpixels.Length; px++)
        {
            rpixels[px] = source.GetPixelBilinear(incX * ((float)px % targetWidth), incY * Mathf.Floor(px / targetWidth));
        }
        result.SetPixels(rpixels, 0);
        result.Apply();
        return result;
    }
    
    public Texture2D FlipTextureVertical(Texture2D original)
    {
        // Flip the texture vertically
        
        Texture2D flipped = new Texture2D(original.width, original.height);

        for (int y = 0; y < original.height; y++)
        {
            for (int x = 0; x < original.width; x++)
            {
                flipped.SetPixel(x, y, original.GetPixel(x, original.height - y - 1));
            }
        }

        flipped.Apply();
        return flipped;
    }
    
    public int[] PrepareImageForInference(Texture2D currentTexture, int defaultSize, int targetSize, int binarizationThreshold)
    {
        // Process drawn image into correct format for network
        
        Color32[] pixels = currentTexture.GetPixels32();
        int imageSize = defaultSize * defaultSize;
        int[] rawInput = new int[imageSize];

        for (int i = 0; i < imageSize; i++)
        {
            rawInput[i] = 255 - pixels[i].r;
        }

        rawInput = ProcessImage(rawInput, defaultSize, defaultSize, targetSize, targetSize);
        rawInput = BinarizeImage(rawInput, binarizationThreshold);
        return rawInput;
    }
}
