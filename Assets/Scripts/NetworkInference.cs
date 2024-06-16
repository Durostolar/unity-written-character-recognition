using System;
using System.Collections;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.Serialization;

public class NetworkInference : MonoBehaviour
{
    // This class handles loading stored neural network and inference in the application,
    // Includes capturing drawing, using the net and updating the UI with prediction results
    
    private Network network;

    [SerializeField] private Draw draw;

    [SerializeField] public SymbolsRecognition task;
    
    [FormerlySerializedAs("symbols")] [SerializeField] private TMP_Text[] symbolTextFields;
    [FormerlySerializedAs("probabilities")] [SerializeField] private TMP_Text[] probabilityTextFields;
    
    private int targetSize = 24;

    public void Start()
    {
        for (int i = 0; i < 5; i++)
        {
            symbolTextFields[i].text = "";
            probabilityTextFields[i].text = "";
        }
        StartCoroutine(DelayedLoad(0.5f));
    }
    
    private IEnumerator DelayedLoad(float delay)
    {
        // Coroutine to wait for a specified delay before loading the network, to make sure that everything is initialized first
        
        yield return new WaitForSeconds(delay);
        LoadNetwork();
    }

    public void LoadNetwork()
    {
        // Load the neural network data from a resource file

        var json =  Resources.Load<TextAsset>("model");
        
        NetworkData data = JsonUtility.FromJson<NetworkData>(json.text);
        network = gameObject.AddComponent<Network>();
        network.LoadData(data);
        
        Debug.Log("Net loaded");
    }

    public void Predict()
    {
        StartCoroutine(UseNetwork());
    }
    
    IEnumerator UseNetwork()
    {
        // Capture the drawing, process the image, run it through the network, and update the UI with the results
        
        yield return StartCoroutine(draw.CaptureDrawing());   
        
        int[] processedInput = ImageTransformations.Transformations.PrepareImageForInference(
            draw.currentTexture, draw.defaultSize, targetSize, 50);
        
        float[,] networkInput = new float[1, targetSize * targetSize];

        for (int i = 0; i < targetSize * targetSize; i++)
        {
            networkInput[0, i] = processedInput[i];
        }
    
        network.ForwardPass(networkInput);

        float[,] outputs = network.Layers[^1].Outputs;
        
        UpdateUI(outputs);
    }
    
    private void UpdateUI(float[,] outputs)
    {
        // Find the top predictions, and update the UI

        float sum = 0;
        float[] scores = new float[outputs.GetLength(1)];
        for (int i = 0; i < scores.Length; i++)
        {
            scores[i] = outputs[0, i];
            sum += scores[i];
        }

        var sortedIndices = Enumerable.Range(0, scores.Length)
            .OrderByDescending(i => scores[i])
            .Take(5).ToList();

        for (int i = 0; i < sortedIndices.Count; i++)
        {
            int index = sortedIndices[i];
            symbolTextFields[i].text = ClassToString(index);
            probabilityTextFields[i].text = (100 * scores[index] / sum).ToString("F2") + "%";
        }
    }
    
    static string ClassToString(int num)
    {
        // Convert a class index to a corresponding string representation (0-9 for digits, A-Z for letters).
        
        if (num < 0 || num > 35)
        {
            throw new ArgumentOutOfRangeException("num", "Integer must be between 0 and 35");
        }

        if (num < 10)
        {
            return num.ToString();
        }
        char c = (char)('A' + (num - 10));
        return c.ToString();
    }
    
    public void EvaluateCollectedTest()
    {
        // Evaulate the loaded network

        if (network == null)
        {
            throw new NullReferenceException("Network has to be loaded first");
        }
        
        task.EvaluateCollectedTest(network);
    }
}
