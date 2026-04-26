using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.XR.ARFoundation;
using UnityEngine.UI;
using System.Collections.Generic;
using System.IO;
using System.Xml.Serialization;

// This lives outside the class so it's accessible and clean
public struct Prediction
{
    public Rect box;
    public float score;
    public int labelIndex;
    public string name;
}

public class ARGPUYOLOInference : MonoBehaviour
{
    [Header("AR Setup")]
    public ARCameraBackground arBackground;
    public RenderTexture destinationRT; // 640x640

    public RawImage displayImage;

    [Header("Model Setup")]
    public ModelAsset modelAsset;
    
    private Worker worker;
    private Tensor<float> inputTensor; // We reuse this every frame

    public float confidenceThreshold = 0.5f;

    [Header("Label Settings")]
    public string[] paintingNames = new string[] 
    {
        "Girl with pearl earing",
        "Guernica",
        "Monalisa",
        "Starry Night",
        "Sunrise",
        "Terrasse Du Cafe",
        "The Kiss",
        "The Liffey Swim",
        "The Persistance of Time",
        "The Scream"
    };

    [Header("Inference Rate")]
    public float inferenceInterval = 0.2f; // 0.2s = 5 FPS. Avoid overloading the GPU with inference every frame.
    private float nextInferenceTime = 0f;
    [Header("UI Display")]
    public GameObject boxPrefab;     // Your box prefab
    public Transform canvasTransform; // The parent Canvas
    private List<GameObject> boxPool = new List<GameObject>();

    [Header("Painting Info")]
    public GameObject infoPanel;           // InfoPanel in the Inspector
    public TMPro.TextMeshProUGUI infoText; // Basic Painting Information displayed 


    [System.Serializable]
    public class Painting {
    public string id;
    public string name;
    public string painter;
    public string year;
    public string description;
    }

    [XmlRoot("Art_InsiderDB")]
    public class PaintingDatabase {
    [XmlElement("painting")]
    public List<Painting> paintings = new List<Painting>();
    }

    private Dictionary<string, Painting> infoLookup = new Dictionary<string, Painting>();

    void Awake() {
        TextAsset xmlFile = Resources.Load<TextAsset>("Art_InsiderDB");
        if (xmlFile != null) {
            XmlSerializer serializer = new XmlSerializer(typeof(PaintingDatabase));
            using (StringReader reader = new StringReader(xmlFile.text)) {
                PaintingDatabase db = (PaintingDatabase)serializer.Deserialize(reader);
                foreach (var p in db.paintings) infoLookup[p.id] = p;
            }
        }
        Debug.Log($"Database Loaded: {infoLookup.Count} paintings found in XML.");
    }
    
    void Start()
    {
        var runtimeModel = ModelLoader.Load(modelAsset);
        worker = new Worker(runtimeModel, BackendType.GPUCompute);

        // Pre-allocate the input tensor once (YOLOv8: 1 batch, 3 channels, 640 height, 640 width)
        inputTensor = new Tensor<float>(new TensorShape(1, 3, 640, 640));
    }

    void Update()
    {
        if (arBackground == null || arBackground.material == null || destinationRT == null) 
            return;

        // 1. Capture AR background to the RenderTexture
        Graphics.Blit(null, destinationRT, arBackground.material);

        //1.5 Display the result on UI Canvas for verification
        if (displayImage != null)
        {
            displayImage.texture = destinationRT;
        }

        // 1.6. ONLY run inference if the timer has expired
        if (Time.time < nextInferenceTime) return;
        // Reset timer only when we actually pass the check
        nextInferenceTime = Time.time + inferenceInterval;

        // --- HEAVY AI LOGIC STARTS HERE ---

        // 2. Copy RT into the existing Tensor
        //    TextureTransform handles resizing/cropping if needed
        TextureConverter.ToTensor(destinationRT, inputTensor, new TextureTransform());

        // 3. Run prediction (Schedule starts the GPU job)
        worker.Schedule(inputTensor);

        // 4. Get the result (PeekOutput retrieves the output tensor by name or index)
        // For YOLOv8, usually the first output (index 0)
        var outputTensor = worker.PeekOutput() as Tensor<float>;
        if (outputTensor == null) return;

        // Move data from GPU to CPU
        float[] data = outputTensor.DownloadToArray();

        List<Prediction> predictions = new List<Prediction>();

        // YOLOv8 uses a "Column-Major" style for this specific tensor shape
        // We iterate through all 8400 candidates
        for (int i = 0; i < 8400; i++)
        {
            float maxScore = 0;
            int labelIndex = -1;

            // Check the 10 label scores (indices 4 to 13)
            for (int c = 4; c < 14; c++)
            {
                // Accessing data[channel * 8400 + candidateIndex]
                float score = data[c * 8400 + i];
                if (score > maxScore)
                {
                    maxScore = score;
                    labelIndex = c - 4;
                }
            }

            // Only keep candidates above our threshold
            if (maxScore > confidenceThreshold)
            {
                string detectedName = paintingNames[labelIndex];
                Debug.Log($"Detected: {detectedName} with {maxScore * 100}% confidence");
                
                float cx = data[0 * 8400 + i];
                float cy = data[1 * 8400 + i];
                float w  = data[2 * 8400 + i];
                float h  = data[3 * 8400 + i];

                // Convert from 640x640 space to 0-1 UV space
                // This makes it easy to scale to any screen size later
                Rect rect = new Rect(
                    (cx - w / 2f) / 640f, 
                    (cy - h / 2f) / 640f, 
                    w / 640f, 
                    h / 640f
                );

                predictions.Add(new Prediction {
                    box = rect,
                    score = maxScore,
                    labelIndex = labelIndex,
                    name = detectedName // You'll need to add 'public string name' to your struct
                });
            }
        }

        // Apply Non-Maximum Suppression (NMS) here to remove duplicate boxes
        var finalBoxes = ApplyNMS(predictions, 0.45f);
        DrawBoxes(finalBoxes); // Update the UI

        // DEBUG: Log how many paintings we found
        if (finalBoxes.Count > 0) Debug.Log($"Found {finalBoxes.Count} paintings!");

        // If only one painting is in the camera frame. User is closer to this painting or show interest in this painting only.
        if (finalBoxes.Count == 1) 
        {
            string label = finalBoxes[0].name; // Get the YOLO name (e.g., "mona_lisa")

            // Look up the "mona_lisa" data in your XML dictionary
            if (infoLookup.TryGetValue(label, out Painting info)) 
            {
                infoPanel.SetActive(true); // Show the big side panel
                
                // This line physically changes the words on your screen!
                infoText.text = $"<b>{info.name}</b>\n" +
                                $"<i>Artist: {info.painter}</i>\n\n" +
                                $"<i>Year: {info.year}</i>\n\n" +
                                $"{info.description}";
            }
        } 
        else 
        {
            // If 0 paintings or 2+ paintings are seen, hide the big info panel
            infoPanel.SetActive(false);
        }
    }

    // public string id;
    // public string name;
    // public string painter;
    // public string year;
    // public string description;

    void OnDestroy()
    {
        worker?.Dispose();
        inputTensor?.Dispose(); // Don't forget to dispose the tensor!
    }

    // Simple NMS to clean up overlapping boxes
    List<Prediction> ApplyNMS(List<Prediction> boxes, float iouThreshold)
    {
        boxes.Sort((a, b) => b.score.CompareTo(a.score));
        List<Prediction> selected = new List<Prediction>();
        List<int> activeIndices = new List<int>();
        for (int i = 0; i < boxes.Count; i++) activeIndices.Add(i);

        while (activeIndices.Count > 0)
        {
            int i = activeIndices[0];
            selected.Add(boxes[i]);
            activeIndices.RemoveAt(0);

            for (int j = activeIndices.Count - 1; j >= 0; j--)
            {
                int k = activeIndices[j];
                if (IntersectionOverUnion(boxes[i].box, boxes[k].box) > iouThreshold)
                    activeIndices.RemoveAt(j);
            }
        }
        return selected;
    }

    float IntersectionOverUnion(Rect a, Rect b)
    {
        float areaA = a.width * a.height;
        float areaB = b.width * b.height;
        float x1 = Mathf.Max(a.x, b.x);
        float y1 = Mathf.Max(a.y, b.y);
        float x2 = Mathf.Min(a.xMax, b.xMax);
        float y2 = Mathf.Min(a.yMax, b.yMax);
        float interArea = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
        return interArea / (areaA + areaB - interArea);
    }
    void DrawBoxes(List<Prediction> finalBoxes)
    {
        // 1. Deactivate all boxes in the pool
        foreach (var b in boxPool) b.SetActive(false);

        // 2. Get Canvas dimensions
        RectTransform canvasRT = canvasTransform.GetComponent<RectTransform>();
        float canvasW = canvasRT.rect.width;
        float canvasH = canvasRT.rect.height;

        for (int i = 0; i < finalBoxes.Count; i++)
        {
            // 3. Get or Create box from pool
            if (i >= boxPool.Count)
            {
                boxPool.Add(Instantiate(boxPrefab, canvasTransform));
            }

            GameObject boxUI = boxPool[i];
            boxUI.SetActive(true);

            // 4. Position and Scale
            RectTransform rt = boxUI.GetComponent<RectTransform>();
            Prediction p = finalBoxes[i];

            // YOLO (0,0) is Top-Left. Unity UI (0,0) is Bottom-Left.
            // We invert the Y: (1 - y_top_corner - height)
            float xPos = p.box.x * canvasW;
            float yPos = (1f - p.box.y - p.box.height) * canvasH;
            float width = p.box.width * canvasW;
            float height = p.box.height * canvasH;

            rt.anchoredPosition = new Vector2(xPos, yPos);
            rt.sizeDelta = new Vector2(width, height);

            // 5. Update Label
            var txt = boxUI.GetComponentInChildren<TMPro.TextMeshProUGUI>();
            if (txt != null) 
                txt.text = $"{p.name} {(p.score * 100):0}%";
        }
    }

}






