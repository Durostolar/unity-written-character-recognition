using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

public class ImageNormalizing : MonoBehaviour
{
    int[] inputImage;

    public void FindBoundingBox(int[] image, int width)
    {
        int upper = 0;
        int lower = 0;
        for (int i = 0; i < image.Length; i++)
        {
            if (image[i] != 0) upper = i/width;
        }
        for (int i = image.Length - 1; i >= 0 ; i--)
        {
            if (image[i] != 0) lower = i/width;
        }

        int j = 0;
        while (j < image.Length)
        {
            j += width;
            if (j > image.Length)
            {
                j += 1 - image.Length;
            }
        }
    }
}
