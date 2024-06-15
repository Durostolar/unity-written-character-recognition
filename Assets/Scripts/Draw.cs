using System.Collections;
using System.Globalization;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;
using Random = UnityEngine.Random;

public class Draw : MonoBehaviour
{
    // Class that allows user to draw lines on a sprite renderer using a LineRenderer component
    // Also handles saving, clearing, and data storage of drawings
    
    [SerializeField] private Camera drawingCamera;
    [SerializeField] private SpriteRenderer targetSpriteRenderer;
    [SerializeField] private Material drawingMaterial;
    private float lineWidth;

    private LineRenderer currentLineRenderer;
    private Vector3? previousPoint;

    public Texture2D currentTexture;

    private float updateTimer;

    private int symbolIndex;
    [SerializeField] private TMP_Text TaskText;

    private Bounds bounds;
    private Vector3 screenMin;
    private Vector3 screenMax;
    private Rect captureZone;

    [SerializeField] private Slider lineSlider;

    public int defaultSize = 28;

    public void Start()
    {
        bounds = targetSpriteRenderer.bounds;
        screenMin = Camera.main.WorldToScreenPoint(bounds.min);
        screenMax = Camera.main.WorldToScreenPoint(bounds.max);
        captureZone = new Rect(screenMin.x, Screen.height - screenMax.y, screenMax.x - screenMin.x, screenMax.y - screenMin.y);
        if (TaskText) CreateTask();
        
        lineSlider.onValueChanged.AddListener(SliderChange);
        lineSlider.value = 0.4f;
        lineWidth = 0.4f;
    }
 
    void Update()
    {
        if (Input.GetMouseButton(0))
        {
            Ray ray = drawingCamera.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;

            if (Physics.Raycast(ray, out hit)) 
            {
                Vector3 currentPoint = hit.point;

                if (previousPoint != null)
                {
                    DrawLine(currentPoint);
                }
                previousPoint = currentPoint;
            }
        }
        else
        {
            previousPoint = null;
            currentLineRenderer = null;
        }
    }
    
    private void SliderChange(float value)
    {
        // Update line width when the slider value changes
        
        lineWidth = lineSlider.value;
    }

    void DrawLine(Vector3 endPoint)
    {
        // Create and draw a line to the specified end point
        
        if (currentLineRenderer == null)
        {
            GameObject lineObject = new GameObject("Line");
            lineObject.transform.SetParent(transform);

            currentLineRenderer = lineObject.AddComponent<LineRenderer>();
            currentLineRenderer.material = drawingMaterial;
            currentLineRenderer.startWidth = lineWidth;
            currentLineRenderer.endWidth = lineWidth;
            currentLineRenderer.positionCount = 0;
        }

        currentLineRenderer.positionCount++;
        currentLineRenderer.SetPosition(currentLineRenderer.positionCount - 1, endPoint);
        currentLineRenderer.startWidth = lineWidth;
        currentLineRenderer.endWidth = lineWidth;
    }

    public void SaveDrawing()
    {
        // Start the coroutine to save the current drawing
        
        StartCoroutine(SaveDrawingRoutine());
    }

    private IEnumerator SaveDrawingRoutine()
    {
        // Capture the drawing, save the data, clear the drawing, and create a new task
        
        yield return StartCoroutine(CaptureDrawing());

        InitiateDataStorage();
        ClearDrawing();
        CreateTask();
    }

    public IEnumerator CaptureDrawing()
    {
        // Capture the current drawing area as a Texture2D and process it
        
        yield return new WaitForEndOfFrame();
        Texture2D screenshot = new Texture2D((int)captureZone.width, (int)captureZone.height, TextureFormat.RGB24, false);

        screenshot.ReadPixels(captureZone, 0, 0);
        screenshot.Apply();
        screenshot = ImageTransformations.Transformations.FlipTextureVertical(screenshot);
        
        currentTexture = ImageTransformations.Transformations.ScaleTexture(screenshot, defaultSize, defaultSize);
    }
    
    public void ClearDrawing()
    {
        // Clear all drawn lines by destroying child objects
        
        foreach (Transform child in transform)
        {
            Destroy(child.gameObject);
        }
    }

    public void InitiateDataStorage()
    {
        // Start the coroutine to send drawings to the server
        
        StartCoroutine(InsertIntoDatabse());
    }

    IEnumerator InsertIntoDatabse()
    {
        WWWForm form = new WWWForm();
        form.AddField("myField", "myData");

        string imageString = "";

        Color32[] pixels = currentTexture.GetPixels32();

        for (int i = 0; i < defaultSize * defaultSize - 1; i++)
        {
            int value = pixels[i].r > 205 ? 0 : 255;
            imageString += value.ToString(CultureInfo.InvariantCulture) + ",";
        }

        imageString += (pixels[^1].r > 205 ? 0 : 255).ToString(CultureInfo.InvariantCulture);

        using (UnityWebRequest www = UnityWebRequest.Post("your request" + imageString, form))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.ConnectionError ||
                www.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.LogError($"Error while sending data: {www.error}");
            }
            else
            {
                Debug.Log("Data successfully sent to the server");
            }
        }
    }

    public void CreateTask()
    {
        // Generate a random drawing task with varying line widths and symbols
        
        lineWidth = Random.Range(0.15f, 0.4f);
        if (lineWidth > 0.3f)
        {
            lineWidth = Random.Range(0.3f, 0.9f);
        }

        symbolIndex = Random.Range(0, 36);

        if (symbolIndex < 10)
        {
            TaskText.text = symbolIndex.ToString();
        }
        else
        {
            TaskText.text = (char)(symbolIndex + 55) + " / " + (char)(symbolIndex + 87);
        }
    }
}

