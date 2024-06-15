using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class DisplayData : MonoBehaviour
{
    // This class handles the loading, processing, and displaying of image data
    
    public RawImage displayImage;

    private Texture2D displayTexture;
    
    List<int[]> datasetList;

    public void ShowImageFromData(int index)
    {
        // Binarize and display image from the dataset at the given index
        
        var image = ImageTransformations.Transformations.BinarizeImage(datasetList[index],50);
        DisplayImage(image,28,28, true);
    }
    
    public void ShowImageFromDataProcessed(int index)
    {
        // Process and display the image from the dataset at the given index, including binarize, resize and rotation
        
        int targetDim = 26;
        var image = ImageTransformations.Transformations.ProcessImage(datasetList[index], 28, 28, targetDim, targetDim);
        image = ImageTransformations.Transformations.RotateImage(image, targetDim, targetDim, 180);
        image = ImageTransformations.Transformations.BinarizeImage(image,50);
        DisplayImage(image,26,26, true);
    }

    public void LoadData(int count)
    {
        // Load the dataset from a CSV file and store the specified number of images
        
        (datasetList, _) = Dataset.dataset.ReadCsv("Assets/Data/my1.csv", 0, count);
        Debug.Log("Loaded " + datasetList.Count + " images");
    }
    
    public void DisplayImage(int[] imageData, int width, int height, bool binarized = false)
    {
        // Display the image data on the UI element, optionally binarize the pixel values
        
        displayTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        Color32[] colors = new Color32[width * height];

        // This is vertically flipped due how unity stores the textures
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = y * width + x;
                
                byte pixelValue = (byte)imageData[index];
                if (binarized) pixelValue *= 255;
                colors[(height - y - 1) * width + x] = new Color32(pixelValue, pixelValue, pixelValue, 255);
            }
        }

        displayTexture.SetPixels32(colors);
        displayTexture.Apply();
        displayImage.texture = displayTexture;
    }
}